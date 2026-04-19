import importlib
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "./whisper_service"
MODELS_DOWNLOAD_ROOT = "./models"

WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
SUPPORTED_TRANSCRIPTION_MODELS = {
    "whisper-1": "turbo",
    "turbo": "turbo",
}

whisperx_models: dict[str, Any] = {}
whisperx_align_models: dict[str, tuple[Any, Any]] = {}
whisperx_diarization_pipeline: Any | None = None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DOWNLOAD_ROOT, exist_ok=True)


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")


class WordTimestamp(BaseModel):
    word: str = Field(description="Recognized word token.")
    start: float | None = Field(
        default=None,
        description="Word start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Word end time in seconds.",
    )
    speaker: str | None = Field(
        default=None,
        description="Speaker label when diarization is enabled.",
    )


class SegmentTimestamp(BaseModel):
    id: int | None = Field(default=None, description="Segment index.")
    start: float | None = Field(
        default=None,
        description="Segment start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Segment end time in seconds.",
    )
    text: str = Field(description="Transcript text for this segment.")
    speaker: str | None = Field(
        default=None,
        description="Speaker label when diarization is enabled.",
    )
    words: list[WordTimestamp] | None = Field(
        default=None,
        description="Per-word timestamps when alignment is available.",
    )


class DiarizationSegment(BaseModel):
    speaker: str | None = Field(
        default=None,
        description="Speaker label such as SPEAKER_00.",
    )
    start: float | None = Field(
        default=None,
        description="Speaker segment start time in seconds.",
    )
    end: float | None = Field(
        default=None,
        description="Speaker segment end time in seconds.",
    )


class TranscriptionSimpleResponse(BaseModel):
    text: str = Field(description="Plain transcription text.")


class TranscriptionAdvancedResponse(TranscriptionSimpleResponse):
    language: str = Field(description="Detected language code.")
    segments: list[SegmentTimestamp] = Field(
        description="Aligned transcript segments with timestamps.",
    )
    diarization: list[DiarizationSegment] = Field(
        description="Speaker diarization segments.",
    )
    speakers: list[str] = Field(
        description="Unique speaker labels that appear in the audio.",
    )


def is_ffmpeg_available() -> bool:
    """Return whether ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def get_whisperx_device() -> str:
    """Choose the runtime device for WhisperX."""
    return os.getenv(
        "WHISPERX_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )


def get_whisperx_compute_type() -> str:
    """Choose WhisperX compute type based on device unless overridden."""
    default = "float16" if get_whisperx_device() == "cuda" else "int8"
    return os.getenv("WHISPERX_COMPUTE_TYPE", default)


def get_diarization_token() -> str | None:
    """Read the Hugging Face token used for WhisperX diarization."""
    return os.getenv("WHISPERX_HF_TOKEN") or os.getenv("HF_TOKEN")


def get_whisperx_module() -> Any:
    """Import WhisperX lazily so tests can patch it easily."""
    try:
        return importlib.import_module("whisperx")
    except ImportError as exc:  # pragma: no cover - exercised in runtime
        raise RuntimeError(
            "whisperx is not installed. Install the whisperx package first."
        ) from exc


def get_whisperx_backend_model_name(model_name: str) -> str:
    """Map API model names to the actual WhisperX backend model name."""
    if model_name not in SUPPORTED_TRANSCRIPTION_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model '{model_name}'. Use one of "
                f"{sorted(SUPPORTED_TRANSCRIPTION_MODELS)}."
            ),
        )

    return SUPPORTED_TRANSCRIPTION_MODELS[model_name]


def load_whisperx_model(model_name: str) -> Any:
    """Load the WhisperX ASR model for the given API model name."""
    whisperx = get_whisperx_module()
    backend_model_name = get_whisperx_backend_model_name(model_name)
    return whisperx.load_model(
        backend_model_name,
        get_whisperx_device(),
        compute_type=get_whisperx_compute_type(),
        download_root=MODELS_DOWNLOAD_ROOT,
    )


def get_whisperx_model(model_name: str) -> Any:
    """Return the cached WhisperX model for the given API model."""
    if model_name not in whisperx_models:
        whisperx_models[model_name] = load_whisperx_model(model_name)
    return whisperx_models[model_name]


def get_align_model(language_code: str) -> tuple[Any, Any]:
    """Return the cached alignment model for a detected language."""
    if language_code not in whisperx_align_models:
        whisperx = get_whisperx_module()
        whisperx_align_models[language_code] = whisperx.load_align_model(
            language_code=language_code,
            device=get_whisperx_device(),
        )
    return whisperx_align_models[language_code]


def load_diarization_pipeline() -> Any:
    """Load the WhisperX diarization pipeline."""
    token = get_diarization_token()
    if not token:
        raise RuntimeError(
            "WhisperX diarization requires WHISPERX_HF_TOKEN or HF_TOKEN."
        )

    diarization_module = importlib.import_module("whisperx.diarize")
    diarization_model_name = os.getenv(
        "WHISPERX_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    return diarization_module.DiarizationPipeline(
        model_name=diarization_model_name,
        token=token,
        device=get_whisperx_device(),
        cache_dir=MODELS_DOWNLOAD_ROOT,
    )


def get_diarization_pipeline() -> Any:
    """Return the cached diarization pipeline, loading it on first use."""
    global whisperx_diarization_pipeline
    if whisperx_diarization_pipeline is None:
        whisperx_diarization_pipeline = load_diarization_pipeline()
    return whisperx_diarization_pipeline


def save_upload_file(file: UploadFile) -> str:
    """Persist an uploaded file and return the temporary path."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    suffix = Path(file.filename).suffix or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return filepath


def cleanup_file(filepath: str) -> None:
    """Remove a temporary file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)


def serialize_diarization_segments(
    diarization_result: Any,
) -> list[DiarizationSegment]:
    """Convert diarization output to typed diarization models."""
    if diarization_result is None:
        return []

    if hasattr(diarization_result, "to_dict"):
        records = diarization_result.to_dict(orient="records")
    elif isinstance(diarization_result, list):
        records = [dict(item) for item in diarization_result]
    else:
        return []

    serialized: list[DiarizationSegment] = []
    for record in records:
        serialized.append(
            DiarizationSegment(
                speaker=record.get("speaker"),
                start=record.get("start"),
                end=record.get("end"),
            )
        )
    return serialized


def serialize_words(words: list[dict[str, Any]] | None) -> list[WordTimestamp] | None:
    """Convert aligned word dictionaries to typed word models."""
    if words is None:
        return None

    return [
        WordTimestamp(
            word=word.get("word", ""),
            start=word.get("start"),
            end=word.get("end"),
            speaker=word.get("speaker"),
        )
        for word in words
    ]


def serialize_segments(segments: list[dict[str, Any]]) -> list[SegmentTimestamp]:
    """Convert aligned segment dictionaries to typed segment models."""
    return [
        SegmentTimestamp(
            id=segment.get("id"),
            start=segment.get("start"),
            end=segment.get("end"),
            text=segment.get("text", ""),
            speaker=segment.get("speaker"),
            words=serialize_words(segment.get("words")),
        )
        for segment in segments
    ]


def build_whisperx_text(segments: list[SegmentTimestamp]) -> str:
    """Join aligned segment text into a single transcript string."""
    parts = [segment.text.strip() for segment in segments]
    return " ".join(part for part in parts if part).strip()


def build_whisperx_response(
    *,
    filepath: str,
    model_name: str,
    language: str | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscriptionAdvancedResponse:
    """Run WhisperX and return the full enriched transcription payload."""
    whisperx = get_whisperx_module()
    audio = whisperx.load_audio(filepath)
    result = get_whisperx_model(model_name).transcribe(
        audio,
        batch_size=WHISPERX_BATCH_SIZE,
        language=language,
    )

    detected_language = result.get("language")
    if not detected_language:
        raise HTTPException(
            status_code=500,
            detail="WhisperX did not return a detected language.",
        )

    model_a, metadata = get_align_model(detected_language)
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        get_whisperx_device(),
        return_char_alignments=False,
    )
    aligned_result["language"] = detected_language

    diarization_segments: list[DiarizationSegment] = []
    if diarize:
        try:
            diarization_result = get_diarization_pipeline()(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        aligned_result = whisperx.assign_word_speakers(
            diarization_result,
            aligned_result,
        )
        diarization_segments = serialize_diarization_segments(diarization_result)

    raw_segments = aligned_result.get("segments", [])
    segments = serialize_segments(raw_segments)
    speakers = sorted(
        {
            speaker
            for speaker in (segment.speaker for segment in segments)
            if speaker
        }
    )

    return TranscriptionAdvancedResponse(
        text=build_whisperx_text(segments),
        language=detected_language,
        segments=segments,
        diarization=diarization_segments,
        speakers=speakers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app

    if not is_ffmpeg_available():
        logger.error("ffmpeg not found in PATH. Please install ffmpeg.")
        raise RuntimeError("ffmpeg not found. Install ffmpeg in the container or host.")

    try:
        yield
    finally:
        global whisperx_diarization_pipeline
        whisperx_models.clear()
        whisperx_diarization_pipeline = None
        whisperx_align_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get(
    "/v1/models",
    summary="List transcription models",
    description=(
        "Returns the model names accepted by `/v1/audio/transcriptions`. "
        "`whisper-1` is kept for compatibility and is routed to WhisperX `turbo`."
    ),
)
async def list_models() -> dict[str, Any]:
    """List available models compatible with OpenAI-style clients."""
    created_at = int(datetime.now().timestamp())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "openai",
                "permission": [],
                "created": created_at,
                "root": get_whisperx_backend_model_name(model_id),
                "parent": None,
            }
            for model_id in SUPPORTED_TRANSCRIPTION_MODELS
        ],
    }


@app.post(
    "/v1/audio/transcriptions",
    summary="Transcribe audio or video",
    description=(
        "Uses WhisperX underneath for all requests. By default it returns only "
        "plain transcription text for compatibility. Set `advanced=true` to "
        "receive aligned timestamps, detected language, and optional speaker "
        "diarization metadata."
    ),
    response_model=TranscriptionSimpleResponse | TranscriptionAdvancedResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The request used an unsupported model name or invalid input.",
        },
        503: {
            "model": ErrorResponse,
            "description": "A required runtime dependency such as diarization credentials is unavailable.",
        },
    },
)
async def transcribe(
    file: UploadFile = File(
        ...,
        description=(
            "Audio or video file to transcribe. Any format supported by ffmpeg "
            "can be uploaded."
        ),
    ),
    model_name: str = Form(
        "whisper-1",
        alias="model",
        description=(
            "Transcription model name. `whisper-1` is a compatibility alias "
            "for WhisperX `turbo`; `turbo` directly selects the same backend."
        ),
    ),
    language: str | None = Form(
        None,
        description=(
            "Optional language code such as `en`, `zh`, or `ja`. When omitted, "
            "WhisperX detects the language automatically."
        ),
    ),
    advanced: bool = Form(
        False,
        description=(
            "When false, return only `{text}`. When true, return full WhisperX "
            "metadata including timestamps, language, and optional diarization."
        ),
    ),
    diarize: bool = Form(
        True,
        description=(
            "Enable speaker diarization in advanced mode. Ignored when "
            "`advanced=false`. Requires `WHISPERX_HF_TOKEN` or `HF_TOKEN`."
        ),
    ),
    min_speakers: int | None = Form(
        None,
        description=(
            "Optional lower bound for the number of speakers. Used only when "
            "`advanced=true` and `diarize=true`."
        ),
    ),
    max_speakers: int | None = Form(
        None,
        description=(
            "Optional upper bound for the number of speakers. Used only when "
            "`advanced=true` and `diarize=true`."
        ),
    ),
) -> TranscriptionSimpleResponse | TranscriptionAdvancedResponse:
    """Transcribe audio using WhisperX behind the legacy endpoint."""
    get_whisperx_backend_model_name(model_name)
    filepath = save_upload_file(file)

    try:
        effective_diarize = advanced and diarize
        response = build_whisperx_response(
            filepath=filepath,
            model_name=model_name,
            language=language,
            diarize=effective_diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if not advanced:
            return TranscriptionSimpleResponse(text=response.text)
        return response
    finally:
        cleanup_file(filepath)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
