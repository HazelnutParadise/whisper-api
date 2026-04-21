"""FastAPI wrapper around the native Transformers Higgs Audio V2 stack."""

from __future__ import annotations

import logging
import gc
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_BACKEND_TTS_MODEL = "eustlb/higgs-audio-v2-generation-3B-base"
PUBLIC_BACKEND_TTS_MODEL = "higgs-audio-v2-generation-3b"
DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_MAX_NEW_TOKENS = 1_024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50

VOICE_PROFILES = {
    "male_en": (
        "Male, American accent, modern speaking rate, moderate-pitch, "
        "friendly tone, and very clear audio."
    ),
    "female_en_story": (
        "She speaks with a calm, gentle, and informative tone at a "
        "measured pace, with excellent articulation and very clear audio. "
        "She naturally brings storytelling to life with an articulate, "
        "genuine, and personable vocal style."
    ),
    "male_en_british": (
        "He speaks with a clear British accent and a conversational, "
        "inquisitive tone. His delivery is articulate and at a moderate "
        "pace, and very clear audio."
    ),
    "female_en_british": (
        "A female voice with a clear British accent speaking at a modern "
        "rate with a moderate-pitch in an expressive and friendly tone and "
        "very clear audio."
    ),
}

VOICE_DESCRIPTIONS = {
    "belinda": (
        "feminine; articulate; expressive; friendly; clear audio; "
        "modern speaking rate"
    ),
    "en_woman": (
        "feminine; warm; natural; expressive; clear audio; modern "
        "speaking rate"
    ),
    "en_man": (
        "masculine; American accent; steady; articulate; clear audio; "
        "modern speaking rate"
    ),
    "broom_salesman": (
        "masculine; lively; persuasive; energetic; clear audio; "
        "slightly theatrical delivery"
    ),
    "mabel": (
        "feminine; soft; thoughtful; calm; clear audio; measured pace"
    ),
    "chadwick": (
        "masculine; low pitch; composed; confident; clear audio; "
        "measured pace"
    ),
}

OPENAI_VOICE_ALIASES = {
    "alloy": "belinda",
    "ash": "en_man",
    "ballad": "mabel",
    "coral": "en_woman",
    "echo": "broom_salesman",
    "fable": "profile:female_en_story",
    "nova": "belinda",
    "onyx": "chadwick",
    "sage": "profile:male_en_british",
    "shimmer": "en_woman",
}

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
    """Loaded Transformers processor/model pair."""

    model: object
    processor: object


class SpeechRequest(BaseModel):
    """OpenAI-style speech request accepted by the native TTS backend."""

    model: str = Field(default=PUBLIC_BACKEND_TTS_MODEL)
    input: str
    voice: str
    instructions: str | None = None
    response_format: str = "pcm"
    speed: float = 1.0
    stream: bool = False
    stream_format: str | None = None


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_engine_from_environment() -> EngineBundle:
    """Load the Higgs processor/model pair with phase-level logging."""
    import torch
    from transformers import (
        AutoProcessor,
        HiggsAudioV2ForConditionalGeneration,
    )

    start_time = time.monotonic()
    model_name = os.environ.get(
        "HIGGS_AUDIO_MODEL",
        DEFAULT_BACKEND_TTS_MODEL,
    )
    device = os.environ.get(
        "HIGGS_AUDIO_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    LOGGER.info(
        "Loading Higgs engine: model=%s device=%s dtype=%s",
        model_name,
        device,
        torch_dtype,
    )

    phase_start = time.monotonic()
    LOGGER.info("Loading Higgs processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    LOGGER.info(
        "Loaded Higgs processor in %.1fs",
        time.monotonic() - phase_start,
    )

    phase_start = time.monotonic()
    LOGGER.info("Loading Higgs model weights...")
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    LOGGER.info(
        "Loaded Higgs model weights in %.1fs",
        time.monotonic() - phase_start,
    )

    phase_start = time.monotonic()
    LOGGER.info("Moving Higgs model to %s...", device)
    model = model.to(device)
    model.eval()
    LOGGER.info(
        "Moved Higgs model to %s in %.1fs",
        device,
        time.monotonic() - phase_start,
    )
    LOGGER.info("Higgs engine ready in %.1fs", time.monotonic() - start_time)
    return EngineBundle(model=model, processor=processor)


def get_engine() -> EngineBundle:
    """Load and cache the Transformers-native Higgs Audio V2 stack."""
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
        LOGGER.exception("Higgs engine load failed.")
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
    """Start model loading in a background thread if needed."""
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
            LOGGER.exception("Higgs engine preload failed.")
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
        name="higgs-engine-preload",
        daemon=True,
    ).start()


def unload_engine() -> None:
    """Drop the cached Higgs model and release CUDA memory."""
    global _ENGINE

    with _ENGINE_COND:
        engine = _ENGINE
        _ENGINE = None
        _ENGINE_COND.notify_all()

    if engine is None:
        return

    LOGGER.info("Unloading Higgs engine after TTS request...")
    del engine
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.exception("Failed to fully clear CUDA cache after Higgs unload.")

    LOGGER.info("Higgs engine unloaded.")


@app.on_event("startup")
def startup_preload() -> None:
    """Optionally start model preload without blocking Uvicorn startup."""
    if env_flag("HIGGS_PRELOAD_ON_STARTUP", default=False):
        preload_engine_async()
    else:
        LOGGER.info("Higgs preload disabled; model will load on first TTS request.")


def resolve_voice_name(voice: str) -> str:
    """Map OpenAI voice aliases to supported Higgs-style descriptions."""
    normalized = voice.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="voice is required.")
    return OPENAI_VOICE_ALIASES.get(normalized, normalized)


def resolve_voice_description(voice: str) -> str:
    """Resolve a voice or profile alias to a scene description string."""
    resolved_voice = resolve_voice_name(voice)

    if resolved_voice.startswith("profile:"):
        profile_name = resolved_voice.split(":", 1)[1]
        speaker_description = VOICE_PROFILES.get(profile_name)
        if speaker_description is None:
            raise HTTPException(
                status_code=400,
                detail=f"voice profile {profile_name!r} is not supported.",
            )
        return speaker_description

    speaker_description = VOICE_DESCRIPTIONS.get(resolved_voice)
    if speaker_description is None:
        raise HTTPException(
            status_code=400,
            detail=f"voice {voice!r} is not supported.",
        )
    return speaker_description


def build_chat_ml_sample(
    input_text: str,
    voice: str,
    instructions: str | None,
) -> list[dict[str, object]]:
    """Convert a simple TTS request into the HF conversation format."""
    scene_items: list[dict[str, str]] = [
        {"type": "text", "text": "Audio is recorded from a quiet room."},
        {"type": "text", "text": f"SPEAKER0: {resolve_voice_description(voice)}"},
    ]
    if instructions:
        scene_items.append({"type": "text", "text": instructions.strip()})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Generate audio following instruction.",
                }
            ],
        },
        {
            "role": "scene",
            "content": scene_items,
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": input_text}],
        },
    ]


def decoded_audio_to_pcm16le(decoded: object, processor: object) -> bytes:
    """Persist decoded audio to WAV, then read frames as PCM16LE bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = handle.name

    try:
        processor.save_audio(decoded, wav_path)
        with wave.open(wav_path, "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise HTTPException(
                    status_code=502,
                    detail="Unexpected multi-channel audio returned by Higgs.",
                )
            if wav_file.getframerate() != DEFAULT_SAMPLING_RATE:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Unexpected Higgs sampling rate "
                        f"{wav_file.getframerate()}; expected "
                        f"{DEFAULT_SAMPLING_RATE}."
                    ),
                )
            if wav_file.getsampwidth() != 2:
                raise HTTPException(
                    status_code=502,
                    detail="Unexpected sample width returned by Higgs.",
                )
            return wav_file.readframes(wav_file.getnframes())
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
            if env_flag("HIGGS_HEALTH_REQUIRE_MODEL", default=False):
                raise HTTPException(
                    status_code=503,
                    detail="Higgs engine is still loading.",
                )
            return {"status": "ok", "engine": "loading"}
        if _ENGINE_ERROR is not None:
            if env_flag("HIGGS_HEALTH_REQUIRE_MODEL", default=False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Higgs engine preload failed: {_ENGINE_ERROR}",
                )
            return {"status": "ok", "engine": "failed"}

    if env_flag("HIGGS_HEALTH_REQUIRE_MODEL", default=False):
        raise HTTPException(
            status_code=503,
            detail="Higgs engine has not started loading yet.",
        )
    return {"status": "ok", "engine": "not_loaded"}


@app.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest):
    """Generate PCM audio with the Transformers-native Higgs Audio V2 path."""
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
            detail="Native Higgs backend only supports response_format=pcm.",
        )
    if payload.stream_format not in (None, "", "audio"):
        raise HTTPException(
            status_code=400,
            detail="stream_format must be omitted or set to `audio`.",
        )

    with _INFERENCE_LOCK:
        try:
            conversation = build_chat_ml_sample(
                payload.input,
                payload.voice,
                payload.instructions,
            )
            bundle = get_engine()
            temperature = float(
                os.environ.get("HIGGS_AUDIO_TEMPERATURE", DEFAULT_TEMPERATURE)
            )
            inputs = bundle.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                sampling_rate=DEFAULT_SAMPLING_RATE,
                return_tensors="pt",
            ).to(bundle.model.device)
            outputs = bundle.model.generate(
                **inputs,
                max_new_tokens=int(
                    os.environ.get(
                        "HIGGS_AUDIO_MAX_NEW_TOKENS",
                        DEFAULT_MAX_NEW_TOKENS,
                    )
                ),
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=float(os.environ.get("HIGGS_AUDIO_TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.environ.get("HIGGS_AUDIO_TOP_K", DEFAULT_TOP_K)),
            )
            decoded = bundle.processor.batch_decode(outputs)
            pcm_bytes = decoded_audio_to_pcm16le(decoded, bundle.processor)
        finally:
            if env_flag("HIGGS_UNLOAD_AFTER_REQUEST", default=True):
                unload_engine()

    if payload.stream:
        return StreamingResponse(iter([pcm_bytes]), media_type="audio/pcm")
    return Response(content=pcm_bytes, media_type="audio/pcm")
