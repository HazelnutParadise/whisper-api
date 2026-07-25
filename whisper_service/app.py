import importlib
import gc
import logging
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from whisper_service.config import (
    MODELS_DOWNLOAD_ROOT,
    SUPPORTED_TRANSCRIPTION_MODELS,
    UPLOAD_FOLDER,
    WHISPERX_AUTO_CHUNK_MIN_SECONDS,
    WHISPERX_BATCH_SIZE,
    WHISPERX_CHUNK_OVERLAP_SECONDS,
    WHISPERX_CHUNK_SECONDS,
    WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM,
    get_diarization_token,
    get_whisperx_compute_type,
    get_whisperx_device,
)
from whisper_service.schemas import (
    DiarizationSegment,
    ErrorResponse,
    SegmentTimestamp,
    TranscriptionAdvancedResponse,
    TranscriptionSimpleResponse,
    WordTimestamp,
)


logger = logging.getLogger(__name__)

whisperx_models: dict[str, Any] = {}
whisperx_align_models: dict[str, tuple[Any, Any]] = {}
whisperx_diarization_pipeline: Any | None = None
runtime_asset_states: dict[str, str] = {}
runtime_asset_errors: dict[str, str] = {}
runtime_asset_events: dict[str, threading.Event] = {}
runtime_asset_lock = threading.Lock()
asr_inference_lock = threading.Lock()


def is_ffmpeg_available() -> bool:
    """Return whether ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


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
            "WhisperX diarization requires HF_TOKEN."
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


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def unload_whisperx_runtime() -> None:
    """Drop cached WhisperX runtime assets and release CUDA memory."""
    global whisperx_diarization_pipeline

    logger.info("Unloading WhisperX runtime assets after transcription...")
    whisperx_models.clear()
    whisperx_align_models.clear()
    whisperx_diarization_pipeline = None

    with runtime_asset_lock:
        runtime_asset_states.clear()
        runtime_asset_errors.clear()
        runtime_asset_events.clear()

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logger.exception("Failed to fully clear CUDA cache after WhisperX unload.")

    logger.info("WhisperX runtime assets unloaded.")


def _mark_runtime_asset_ready(resource: str) -> None:
    """Mark a runtime model asset as available."""
    with runtime_asset_lock:
        runtime_asset_states[resource] = "ready"
        runtime_asset_errors.pop(resource, None)
        event = runtime_asset_events.get(resource)
        if event is not None:
            event.set()


def _mark_runtime_asset_failed(resource: str, exc: Exception) -> None:
    """Mark a runtime model asset download as failed."""
    with runtime_asset_lock:
        runtime_asset_states[resource] = "failed"
        runtime_asset_errors[resource] = str(exc)
        event = runtime_asset_events.get(resource)
        if event is not None:
            event.set()


def _load_runtime_asset(resource: str, loader: Any) -> None:
    """Load a runtime asset and update synchronized state."""
    try:
        loader()
    except Exception as exc:
        logger.exception("Runtime asset load failed: %s", resource)
        _mark_runtime_asset_failed(resource, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load runtime asset '{resource}': {exc}",
        ) from exc

    _mark_runtime_asset_ready(resource)


def ensure_runtime_asset(
    *,
    resource: str,
    is_ready: Any,
    loader: Any,
) -> None:
    """Ensure an asset is ready, using single-flight loading across requests."""
    while True:
        if is_ready():
            _mark_runtime_asset_ready(resource)
            return

        run_loader = False
        event: threading.Event | None = None
        with runtime_asset_lock:
            state = runtime_asset_states.get(resource)
            if state == "ready":
                return

            if state == "running":
                event = runtime_asset_events.get(resource)
                if event is None:
                    event = threading.Event()
                    runtime_asset_events[resource] = event
            else:
                runtime_asset_states[resource] = "running"
                runtime_asset_errors.pop(resource, None)
                event = threading.Event()
                runtime_asset_events[resource] = event
                run_loader = True

        if run_loader:
            _load_runtime_asset(resource, loader)
            return

        if event is None:
            continue
        event.wait()

        with runtime_asset_lock:
            state = runtime_asset_states.get(resource)
            if state == "ready":
                return
            if state == "failed":
                error = runtime_asset_errors.get(resource, "unknown error")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load runtime asset '{resource}': {error}",
                )


def get_diarization_resource_name() -> str:
    """Return the configured diarization resource identifier."""
    return os.getenv(
        "WHISPERX_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    )


def ensure_runtime_assets_ready(
    *,
    model_name: str,
    language: str | None,
    diarize: bool,
) -> None:
    """Ensure required runtime assets are loaded before transcription."""
    backend_model_name = get_whisperx_backend_model_name(model_name)

    ensure_runtime_asset(
        resource=f"asr:{backend_model_name}",
        is_ready=lambda: model_name in whisperx_models,
        loader=lambda: get_whisperx_model(model_name),
    )

    if language:
        ensure_runtime_asset(
            resource=f"align:{language}",
            is_ready=lambda: language in whisperx_align_models,
            loader=lambda: get_align_model(language),
        )

    if diarize:
        if not get_diarization_token():
                raise HTTPException(
                    status_code=503,
                    detail="WhisperX diarization requires HF_TOKEN.",
                )

        ensure_runtime_asset(
            resource=f"diarization:{get_diarization_resource_name()}",
            is_ready=lambda: whisperx_diarization_pipeline is not None,
            loader=get_diarization_pipeline,
        )

    return None


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


def build_simple_transcription_response(
    *,
    filepath: str,
    model_name: str,
    language: str | None,
) -> TranscriptionSimpleResponse:
    """Run WhisperX ASR only and return plain transcript text."""
    whisperx = get_whisperx_module()
    audio = whisperx.load_audio(filepath)
    result = get_whisperx_model(model_name).transcribe(
        audio,
        batch_size=WHISPERX_BATCH_SIZE,
        language=language,
    )
    raw_segments = result.get("segments", [])
    text = " ".join(
        segment.get("text", "").strip()
        for segment in raw_segments
        if segment.get("text")
    ).strip()
    return TranscriptionSimpleResponse(text=text)


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

    ensure_runtime_assets_ready(
        model_name=model_name,
        language=detected_language,
        diarize=diarize,
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


def should_use_chunked_transcription(filepath: str) -> bool:
    """Return whether the upload should use chunked transcription."""
    if WHISPERX_AUTO_CHUNK_MIN_SECONDS <= 0:
        return False

    try:
        duration_seconds = get_audio_duration_seconds(filepath)
    except HTTPException:
        # ffprobe could not read the container, but WhisperX may still decode
        # it. Fall back to the single-pass path rather than failing outright.
        return False

    return duration_seconds >= WHISPERX_AUTO_CHUNK_MIN_SECONDS


def get_audio_duration_seconds(filepath: str) -> float:
    """Read audio duration in seconds with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filepath,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to inspect audio duration with ffprobe.",
        )

    output = result.stdout.strip()
    if not output:
        raise HTTPException(
            status_code=500,
            detail="Audio duration is unavailable.",
        )

    try:
        duration = float(output)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Audio duration returned by ffprobe is invalid.",
        ) from exc

    return max(duration, 0.0)


def generate_chunk_specs(duration_seconds: float) -> list[dict[str, float | int]]:
    """Create chunk metadata for ffmpeg slicing and merge offsets."""
    if duration_seconds <= 0:
        return [{"index": 0, "nominal_start": 0.0, "extract_start": 0.0, "extract_duration": 0.0}]

    chunk_seconds = max(1, WHISPERX_CHUNK_SECONDS)
    overlap = max(0.0, WHISPERX_CHUNK_OVERLAP_SECONDS)

    specs: list[dict[str, float | int]] = []
    index = 0
    nominal_start = 0.0
    while nominal_start < duration_seconds:
        extract_start = max(0.0, nominal_start - overlap)
        extract_duration = min(chunk_seconds + (overlap if index > 0 else 0.0), duration_seconds - extract_start)
        specs.append(
            {
                "index": index,
                "nominal_start": nominal_start,
                "extract_start": extract_start,
                "extract_duration": extract_duration,
            }
        )
        index += 1
        nominal_start += chunk_seconds

    return specs


def extract_audio_chunk(filepath: str, spec: dict[str, float | int]) -> str:
    """Extract one chunk from the original audio file."""
    chunk_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4().hex}_chunk_{int(spec['index'])}.wav",
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{float(spec['extract_start']):.3f}",
            "-i",
            filepath,
            "-t",
            f"{float(spec['extract_duration']):.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            chunk_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        cleanup_file(chunk_path)
        raise HTTPException(status_code=500, detail="Failed to split audio into chunks.")
    return chunk_path


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return overlap duration of two time ranges."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def build_chunk_speaker_mapping(
    *,
    previous_segments: list[SegmentTimestamp],
    current_segments: list[SegmentTimestamp],
    boundary_start: float,
    overlap_seconds: float,
    known_speakers: list[str],
    min_similarity: float,
) -> dict[str, str]:
    """Map current chunk speaker labels to global labels using boundary overlap."""
    window_start = max(0.0, boundary_start - overlap_seconds)
    window_end = boundary_start
    min_similarity = max(0.0, min(min_similarity, 1.0))

    curr_coverage: dict[str, float] = {}
    prev_coverage: dict[str, float] = {}

    for curr in current_segments:
        if not curr.speaker or curr.start is None or curr.end is None:
            continue
        curr_start = max(curr.start, window_start)
        curr_end = min(curr.end, window_end)
        if curr_end <= curr_start:
            continue
        curr_coverage[curr.speaker] = curr_coverage.get(curr.speaker, 0.0) + (
            curr_end - curr_start
        )

    # Build similarity by temporal intersection inside overlap window.
    pair_scores: dict[tuple[str, str], float] = {}
    for prev in previous_segments:
        if not prev.speaker or prev.start is None or prev.end is None:
            continue
        prev_start = max(prev.start, window_start)
        prev_end = min(prev.end, window_end)
        if prev_end <= prev_start:
            continue
        prev_coverage[prev.speaker] = prev_coverage.get(prev.speaker, 0.0) + (
            prev_end - prev_start
        )

        for curr in current_segments:
            if not curr.speaker or curr.start is None or curr.end is None:
                continue
            curr_start = max(curr.start, window_start)
            curr_end = min(curr.end, window_end)
            if curr_end <= curr_start:
                continue

            score = _overlap_duration(prev_start, prev_end, curr_start, curr_end)
            if score <= 0:
                continue
            key = (curr.speaker, prev.speaker)
            pair_scores[key] = pair_scores.get(key, 0.0) + score

    pair_similarity: dict[tuple[str, str], float] = {}
    for (curr_label, prev_label), overlap in pair_scores.items():
        curr_total = curr_coverage.get(curr_label, 0.0)
        prev_total = prev_coverage.get(prev_label, 0.0)
        if curr_total <= 0 or prev_total <= 0:
            continue
        pair_similarity[(curr_label, prev_label)] = overlap / max(curr_total, prev_total)

    mapping: dict[str, str] = {}
    used_global: set[str] = set()
    for (curr_label, prev_label), similarity in sorted(
        pair_similarity.items(), key=lambda item: item[1], reverse=True
    ):
        if curr_label in mapping or prev_label in used_global:
            continue
        if similarity < min_similarity:
            continue
        mapping[curr_label] = prev_label
        used_global.add(prev_label)

    existing = sorted({speaker for speaker in known_speakers if speaker})
    next_idx = 0
    while f"SPEAKER_{next_idx:02d}" in existing:
        next_idx += 1

    for segment in current_segments:
        curr_label = segment.speaker
        if not curr_label or curr_label in mapping:
            continue
        global_label = f"SPEAKER_{next_idx:02d}"
        mapping[curr_label] = global_label
        existing.append(global_label)
        next_idx += 1

    return mapping


def apply_speaker_mapping_to_segments(
    segments: list[SegmentTimestamp], mapping: dict[str, str]
) -> None:
    """Rewrite segment and word speaker labels in place."""
    for segment in segments:
        if segment.speaker in mapping:
            segment.speaker = mapping[segment.speaker]
        if not segment.words:
            continue
        for word in segment.words:
            if word.speaker in mapping:
                word.speaker = mapping[word.speaker]


def apply_speaker_mapping_to_diarization(
    diarization: list[DiarizationSegment], mapping: dict[str, str]
) -> None:
    """Rewrite diarization speaker labels in place."""
    for diarization_segment in diarization:
        if diarization_segment.speaker in mapping:
            diarization_segment.speaker = mapping[diarization_segment.speaker]


def shift_advanced_response_timestamps(
    payload: TranscriptionAdvancedResponse,
    *,
    offset_seconds: float,
) -> None:
    """Shift all response timestamps by offset."""
    for segment in payload.segments:
        if segment.start is not None:
            segment.start += offset_seconds
        if segment.end is not None:
            segment.end += offset_seconds
        if not segment.words:
            continue
        for word in segment.words:
            if word.start is not None:
                word.start += offset_seconds
            if word.end is not None:
                word.end += offset_seconds

    for diarization_segment in payload.diarization:
        if diarization_segment.start is not None:
            diarization_segment.start += offset_seconds
        if diarization_segment.end is not None:
            diarization_segment.end += offset_seconds


def filter_chunk_overlap_prefix(
    payload: TranscriptionAdvancedResponse,
    *,
    chunk_nominal_start: float,
) -> None:
    """Drop overlapped prefix content from a chunk to reduce duplicates."""
    if chunk_nominal_start <= 0:
        return

    payload.segments = [
        segment
        for segment in payload.segments
        if segment.end is None or segment.end > chunk_nominal_start
    ]
    payload.diarization = [
        segment
        for segment in payload.diarization
        if segment.end is None or segment.end > chunk_nominal_start
    ]


def build_chunked_whisperx_response(
    *,
    filepath: str,
    model_name: str,
    language: str | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscriptionAdvancedResponse:
    """Run chunked transcription and merge timestamps and speakers globally."""
    duration_seconds = get_audio_duration_seconds(filepath)
    chunk_specs = generate_chunk_specs(duration_seconds)
    if len(chunk_specs) <= 1:
        return build_whisperx_response(
            filepath=filepath,
            model_name=model_name,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    merged_segments: list[SegmentTimestamp] = []
    merged_diarization: list[DiarizationSegment] = []
    merged_language: str | None = language
    known_speakers: list[str] = []
    chunk_paths: list[str] = []

    try:
        for spec in chunk_specs:
            chunk_path = extract_audio_chunk(filepath, spec)
            chunk_paths.append(chunk_path)

            chunk_result = build_whisperx_response(
                filepath=chunk_path,
                model_name=model_name,
                language=language,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            shift_advanced_response_timestamps(
                chunk_result,
                offset_seconds=float(spec["extract_start"]),
            )
            filter_chunk_overlap_prefix(
                chunk_result,
                chunk_nominal_start=float(spec["nominal_start"]),
            )

            if diarize:
                mapping = build_chunk_speaker_mapping(
                    previous_segments=merged_segments,
                    current_segments=chunk_result.segments,
                    boundary_start=float(spec["nominal_start"]),
                    overlap_seconds=max(0.0, WHISPERX_CHUNK_OVERLAP_SECONDS),
                    known_speakers=known_speakers,
                    min_similarity=WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM,
                )
                apply_speaker_mapping_to_segments(chunk_result.segments, mapping)
                apply_speaker_mapping_to_diarization(chunk_result.diarization, mapping)

            merged_segments.extend(chunk_result.segments)
            merged_diarization.extend(chunk_result.diarization)
            if not merged_language:
                merged_language = chunk_result.language

            for speaker in chunk_result.speakers:
                if speaker and speaker not in known_speakers:
                    known_speakers.append(speaker)

        speakers = sorted(
            {
                speaker
                for speaker in (segment.speaker for segment in merged_segments)
                if speaker
            }
        )
        return TranscriptionAdvancedResponse(
            text=build_whisperx_text(merged_segments),
            language=merged_language or "unknown",
            segments=merged_segments,
            diarization=merged_diarization,
            speakers=speakers,
        )
    finally:
        for chunk_path in chunk_paths:
            cleanup_file(chunk_path)


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
            "`advanced=false`. Requires `HF_TOKEN`."
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
    effective_diarize = advanced and diarize
    filepath = save_upload_file(file)

    with asr_inference_lock:
        try:
            ensure_runtime_assets_ready(
                model_name=model_name,
                language=language if advanced else None,
                diarize=effective_diarize,
            )

            if advanced:
                if should_use_chunked_transcription(filepath):
                    response = build_chunked_whisperx_response(
                        filepath=filepath,
                        model_name=model_name,
                        language=language,
                        diarize=effective_diarize,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                else:
                    response = build_whisperx_response(
                        filepath=filepath,
                        model_name=model_name,
                        language=language,
                        diarize=effective_diarize,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
            else:
                response = build_simple_transcription_response(
                    filepath=filepath,
                    model_name=model_name,
                    language=language,
                )
            return response
        finally:
            cleanup_file(filepath)
            if env_flag("WHISPERX_UNLOAD_AFTER_REQUEST", default=True):
                unload_whisperx_runtime()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
