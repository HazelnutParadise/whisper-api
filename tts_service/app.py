"""FastAPI wrapper around Coqui TTS with an OpenAI-style speech shape."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_BACKEND_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
PUBLIC_BACKEND_TTS_MODEL = "coqui-tts"
DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_COQUI_LANGUAGE = "zh-cn"
DEFAULT_SPEAKER_WAV_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
DEFAULT_SPEAKER_WAV_PATH = "/root/.local/share/tts/default-speaker.flac"

SUPPORTED_TTS_MODELS = {
    "",
    "tts-1",
    "tts-1-hd",
    PUBLIC_BACKEND_TTS_MODEL,
    DEFAULT_BACKEND_TTS_MODEL,
}

_ENGINE = None
_ENGINE_ERROR = None
_ENGINE_LOADING = False
_ENGINE_COND = threading.Condition()
_INFERENCE_LOCK = threading.Lock()

app = FastAPI()
LOGGER = logging.getLogger("uvicorn.error")


@dataclass
class EngineBundle:
    """Loaded Coqui TTS model."""

    model: object


class SpeechRequest(BaseModel):
    """OpenAI-style speech request accepted by the Coqui backend."""

    model: str = Field(default=PUBLIC_BACKEND_TTS_MODEL)
    input: str
    voice: str
    instructions: str | None = None
    response_format: str = "pcm"
    speed: float = 1.0
    stream: bool = False
    stream_format: str | None = None


def clear_cuda_cache() -> None:
    """Release unreferenced CUDA allocations after failures or unloads."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.exception("Failed to clear CUDA cache.")


def is_cuda_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM without importing torch at module import time."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "outofmemoryerror" in message


def allow_coqui_xtts_checkpoint_globals() -> None:
    """Allow PyTorch 2.6+ to load trusted Coqui XTTS checkpoint config objects."""
    import torch
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

    torch.serialization.add_safe_globals(
        [
            BaseDatasetConfig,
            XttsArgs,
            XttsAudioConfig,
            XttsConfig,
        ]
    )


def ensure_default_speaker_wav() -> str:
    """Download the default XTTS reference voice sample into the mounted TTS cache."""
    if os.path.exists(DEFAULT_SPEAKER_WAV_PATH):
        return DEFAULT_SPEAKER_WAV_PATH

    os.makedirs(os.path.dirname(DEFAULT_SPEAKER_WAV_PATH), exist_ok=True)
    LOGGER.info("Downloading default Coqui speaker reference to %s", DEFAULT_SPEAKER_WAV_PATH)
    urllib.request.urlretrieve(DEFAULT_SPEAKER_WAV_URL, DEFAULT_SPEAKER_WAV_PATH)
    return DEFAULT_SPEAKER_WAV_PATH


def load_engine_from_environment() -> EngineBundle:
    """Load the Coqui TTS model with phase-level logging."""
    import torch
    from TTS.api import TTS

    allow_coqui_xtts_checkpoint_globals()
    ensure_default_speaker_wav()

    start_time = time.monotonic()
    model_name = DEFAULT_BACKEND_TTS_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested for Coqui TTS but unavailable; falling back to CPU.")
        device = "cpu"

    LOGGER.info("Loading Coqui TTS engine: model=%s device=%s", model_name, device)
    phase_start = time.monotonic()
    model = TTS(model_name=model_name, progress_bar=False)
    model.to(device)
    LOGGER.info("Loaded Coqui TTS model in %.1fs", time.monotonic() - phase_start)
    LOGGER.info("Coqui TTS engine ready in %.1fs", time.monotonic() - start_time)
    return EngineBundle(model=model)


def get_engine() -> EngineBundle:
    """Load and cache the Coqui TTS model."""
    global _ENGINE
    global _ENGINE_ERROR
    global _ENGINE_LOADING

    with _ENGINE_COND:
        if _ENGINE is not None:
            return _ENGINE

        while _ENGINE_LOADING:
            _ENGINE_COND.wait()
            if _ENGINE is not None:
                return _ENGINE

        _ENGINE_LOADING = True
        _ENGINE_ERROR = None

    try:
        engine = load_engine_from_environment()
    except Exception as exc:
        LOGGER.exception("Coqui TTS engine load failed.")
        with _ENGINE_COND:
            _ENGINE_LOADING = False
            _ENGINE_ERROR = exc
            _ENGINE_COND.notify_all()
        raise

    with _ENGINE_COND:
        _ENGINE = engine
        _ENGINE_LOADING = False
        _ENGINE_ERROR = None
        _ENGINE_COND.notify_all()
        return _ENGINE


def preload_engine_async() -> None:
    """Start model loading in a background thread if configured."""
    global _ENGINE_LOADING
    global _ENGINE_ERROR

    with _ENGINE_COND:
        if _ENGINE is not None or _ENGINE_LOADING:
            return
        _ENGINE_LOADING = True
        _ENGINE_ERROR = None

    def _worker() -> None:
        global _ENGINE
        global _ENGINE_ERROR
        global _ENGINE_LOADING

        try:
            engine = load_engine_from_environment()
        except Exception as exc:
            LOGGER.exception("Coqui TTS engine preload failed.")
            with _ENGINE_COND:
                _ENGINE_LOADING = False
                _ENGINE_ERROR = exc
                _ENGINE_COND.notify_all()
            return

        with _ENGINE_COND:
            _ENGINE = engine
            _ENGINE_LOADING = False
            _ENGINE_ERROR = None
            _ENGINE_COND.notify_all()

    threading.Thread(
        target=_worker,
        name="coqui-tts-preload",
        daemon=True,
    ).start()


def unload_engine() -> None:
    """Drop the cached Coqui model and release CUDA memory."""
    global _ENGINE

    with _ENGINE_COND:
        engine = _ENGINE
        _ENGINE = None
        _ENGINE_COND.notify_all()

    if engine is None:
        return

    LOGGER.info("Unloading Coqui TTS engine after request...")
    del engine
    clear_cuda_cache()
    LOGGER.info("Coqui TTS engine unloaded.")


def cleanup_request_runtime_refs(runtime_refs: dict[str, object]) -> None:
    """Drop per-request model/tensor references before CUDA cache cleanup."""
    runtime_refs.clear()
    clear_cuda_cache()


@app.on_event("startup")
def startup_preload() -> None:
    """Keep startup cheap; the model loads on the first TTS request."""
    LOGGER.info("Coqui TTS model will load on first request.")


def coqui_tts_kwargs(payload: SpeechRequest, model: object) -> dict[str, object]:
    """Build optional Coqui synthesis kwargs for models that need them."""
    kwargs: dict[str, object] = {}

    if not kwargs.get("speaker") and getattr(model, "is_multi_speaker", False):
        speakers = getattr(model, "speakers", None) or []
        if speakers:
            kwargs["speaker"] = payload.voice if payload.voice in speakers else speakers[0]
        else:
            kwargs["speaker_wav"] = ensure_default_speaker_wav()

    if getattr(model, "is_multi_lingual", False):
        kwargs["language"] = DEFAULT_COQUI_LANGUAGE
        languages = getattr(model, "languages", None) or []
        if not kwargs.get("language") and languages:
            kwargs["language"] = languages[0]

    if getattr(model, "is_multi_speaker", False) and not (
        kwargs.get("speaker") or kwargs.get("speaker_wav")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected Coqui model requires a speaker, but no built-in "
                "speaker is available."
            ),
        )
    if getattr(model, "is_multi_lingual", False) and not kwargs.get("language"):
        raise HTTPException(
            status_code=400,
            detail="The selected Coqui model requires a language.",
        )

    return kwargs


def wav_file_to_pcm16le(wav_path: str) -> bytes:
    """Convert a Coqui WAV file to the gateway's expected PCM16LE format."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            wav_path,
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLING_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert Coqui audio to PCM: {result.stderr.decode(errors='ignore')}",
        )
    return result.stdout


def synthesize_to_pcm(payload: SpeechRequest, bundle: EngineBundle) -> bytes:
    """Run Coqui synthesis and return PCM16LE bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = handle.name

    try:
        kwargs = coqui_tts_kwargs(payload, bundle.model)
        bundle.model.tts_to_file(
            text=payload.input,
            file_path=wav_path,
            **kwargs,
        )
        return wav_file_to_pcm16le(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Report service readiness without triggering blocking model loads."""
    with _ENGINE_COND:
        if _ENGINE is not None:
            return {"status": "ok", "engine": "ready"}
        if _ENGINE_LOADING:
            return {"status": "ok", "engine": "loading"}
        if _ENGINE_ERROR is not None:
            return {"status": "ok", "engine": "failed"}

    return {"status": "ok", "engine": "not_loaded"}


@app.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest):
    """Generate PCM audio with Coqui TTS."""
    if payload.model not in SUPPORTED_TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                "model must be one of "
                "`tts-1`, `tts-1-hd`, or "
                f"`{PUBLIC_BACKEND_TTS_MODEL}`."
            ),
        )
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input is required.")
    if payload.response_format != "pcm":
        raise HTTPException(
            status_code=400,
            detail="Coqui backend only supports response_format=pcm.",
        )
    if payload.stream_format not in (None, "", "audio"):
        raise HTTPException(
            status_code=400,
            detail="stream_format must be omitted or set to `audio`.",
        )

    runtime_refs: dict[str, object] = {}

    try:
        with _INFERENCE_LOCK:
            runtime_refs["bundle"] = get_engine()
            pcm_bytes = synthesize_to_pcm(payload, runtime_refs["bundle"])
            cleanup_request_runtime_refs(runtime_refs)
            unload_engine()
    except HTTPException:
        cleanup_request_runtime_refs(runtime_refs)
        unload_engine()
        raise
    except Exception as exc:
        cleanup_request_runtime_refs(runtime_refs)
        unload_engine()
        if is_cuda_oom(exc):
            LOGGER.exception("Coqui TTS generation failed because CUDA memory is exhausted.")
            raise HTTPException(
                status_code=503,
                detail=(
                    "TTS backend ran out of GPU memory while loading or "
                    "generating Coqui speech. Retry after GPU memory is freed "
                    "or place ASR and TTS on separate GPUs."
                ),
            ) from exc
        LOGGER.exception("Coqui TTS generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Coqui TTS generation failed: {exc}",
        ) from exc

    if payload.stream:
        return StreamingResponse(
            iter([pcm_bytes]),
            media_type="audio/pcm",
        )
    return Response(
        content=pcm_bytes,
        media_type="audio/pcm",
    )
