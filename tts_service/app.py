"""FastAPI wrapper around the native Higgs Audio Python engine."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_BACKEND_TTS_MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"
DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_MAX_NEW_TOKENS = 1_024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

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
    DEFAULT_BACKEND_TTS_MODEL,
}

VOICE_PROMPTS_DIR = Path(
    os.environ.get(
        "HIGGS_AUDIO_VOICE_PROMPTS_DIR",
        "/opt/higgs-audio/examples/voice_prompts",
    )
)

_ENGINE = None
_ENGINE_LOCK = threading.Lock()

app = FastAPI()


class SpeechRequest(BaseModel):
    """OpenAI-style speech request accepted by the native TTS backend."""

    model: str = Field(default=DEFAULT_BACKEND_TTS_MODEL)
    input: str
    voice: str
    instructions: str | None = None
    response_format: str = "pcm"
    speed: float = 1.0
    stream: bool = False
    stream_format: str | None = None


def get_engine():
    """Load and cache the native HiggsAudioServeEngine."""
    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

        model_name = os.environ.get(
            "HIGGS_AUDIO_MODEL",
            DEFAULT_BACKEND_TTS_MODEL,
        )
        audio_tokenizer = os.environ.get(
            "HIGGS_AUDIO_TOKENIZER",
            DEFAULT_AUDIO_TOKENIZER,
        )
        device = os.environ.get("HIGGS_AUDIO_DEVICE", "cuda")
        _ENGINE = HiggsAudioServeEngine(
            model_name,
            audio_tokenizer,
            device=device,
        )
        return _ENGINE


def resolve_voice_name(voice: str) -> str:
    """Map OpenAI voice aliases to a Higgs prompt voice or profile."""
    normalized = voice.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="voice is required.")
    return OPENAI_VOICE_ALIASES.get(normalized, normalized)


def build_system_prompt(
    instructions: str | None = None,
    speaker_description: str | None = None,
) -> str:
    """Build a simple scene description for native Higgs generation."""
    lines = [
        "Generate audio following instruction.",
        "",
        "<|scene_desc_start|>",
        "Audio is recorded from a quiet room.",
    ]

    if speaker_description:
        lines.append(f"SPEAKER0: {speaker_description}")

    if instructions:
        lines.append(instructions.strip())

    lines.append("<|scene_desc_end|>")
    return "\n".join(lines)


def build_chat_ml_sample(input_text: str, voice: str, instructions: str | None):
    """Convert a simple TTS request into the ChatML format expected by Higgs."""
    resolved_voice = resolve_voice_name(voice)
    system_prompt = build_system_prompt(instructions=instructions)

    if resolved_voice.startswith("profile:"):
        profile_name = resolved_voice.split(":", 1)[1]
        speaker_description = VOICE_PROFILES.get(profile_name)
        if speaker_description is None:
            raise HTTPException(
                status_code=400,
                detail=f"voice profile {profile_name!r} is not supported.",
            )
        from boson_multimodal.data_types import ChatMLSample, Message

        return ChatMLSample(
            messages=[
                Message(
                    role="system",
                    content=build_system_prompt(
                        instructions=instructions,
                        speaker_description=speaker_description,
                    ),
                ),
                Message(role="user", content=input_text),
            ]
        )

    prompt_text_path = VOICE_PROMPTS_DIR / f"{resolved_voice}.txt"
    prompt_audio_path = VOICE_PROMPTS_DIR / f"{resolved_voice}.wav"
    if not prompt_text_path.is_file() or not prompt_audio_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"voice {voice!r} is not supported.",
        )

    from boson_multimodal.data_types import AudioContent, ChatMLSample, Message

    messages = []
    if instructions:
        messages.append(Message(role="system", content=system_prompt))

    messages.extend(
        [
            Message(role="user", content=prompt_text_path.read_text(encoding="utf-8")),
            Message(
                role="assistant",
                content=AudioContent(audio_url=str(prompt_audio_path)),
            ),
            Message(role="user", content=input_text),
        ]
    )
    return ChatMLSample(messages=messages)


def audio_to_pcm16le(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Convert generated mono audio into little-endian 16-bit PCM bytes."""
    if audio is None or audio.size == 0:
        raise HTTPException(status_code=502, detail="Higgs backend returned no audio.")
    if sampling_rate != DEFAULT_SAMPLING_RATE:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unexpected Higgs sampling rate "
                f"{sampling_rate}; expected {DEFAULT_SAMPLING_RATE}."
            ),
        )

    audio_1d = np.asarray(audio).reshape(-1)
    pcm = np.clip(audio_1d, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    return pcm.tobytes()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Report service readiness once the native engine has loaded."""
    get_engine()
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest):
    """Generate PCM audio with the native Higgs Audio Python engine."""
    if payload.model not in SUPPORTED_TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                "model must be one of "
                "`tts-1`, `tts-1-hd`, or "
                f"`{DEFAULT_BACKEND_TTS_MODEL}`."
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

    sample = build_chat_ml_sample(payload.input, payload.voice, payload.instructions)
    output = get_engine().generate(
        chat_ml_sample=sample,
        max_new_tokens=int(
            os.environ.get("HIGGS_AUDIO_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
        ),
        temperature=float(
            os.environ.get("HIGGS_AUDIO_TEMPERATURE", DEFAULT_TEMPERATURE)
        ),
        top_p=float(os.environ.get("HIGGS_AUDIO_TOP_P", DEFAULT_TOP_P)),
        top_k=int(os.environ.get("HIGGS_AUDIO_TOP_K", DEFAULT_TOP_K)),
        stop_strings=DEFAULT_STOP_STRINGS,
        force_audio_gen=True,
    )
    pcm_bytes = audio_to_pcm16le(output.audio, output.sampling_rate)

    if payload.stream:
        return StreamingResponse(iter([pcm_bytes]), media_type="audio/pcm")
    return Response(content=pcm_bytes, media_type="audio/pcm")
