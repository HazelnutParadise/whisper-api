import os

import torch


UPLOAD_FOLDER = "/app/whisper_service"
MODELS_DOWNLOAD_ROOT = "/app/models"

WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
WHISPERX_CHUNK_SECONDS = int(os.getenv("WHISPERX_CHUNK_SECONDS", "600"))
WHISPERX_CHUNK_OVERLAP_SECONDS = float(
    os.getenv("WHISPERX_CHUNK_OVERLAP_SECONDS", "3")
)
# Chunking is keyed on audio duration, not upload size: a 16 kHz mono mp3 runs
# about 29 MB per hour while lossless audio of the same length runs past 500 MB,
# so a byte threshold splits the two formats at wildly different lengths.
WHISPERX_AUTO_CHUNK_MIN_SECONDS = float(
    os.getenv("WHISPERX_AUTO_CHUNK_MIN_SECONDS", "3600")
)
WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM = float(
    os.getenv("WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM", "0.7")
)
SUPPORTED_TRANSCRIPTION_MODELS = {
    "whisper-1": "turbo",
    "turbo": "turbo",
}

HF_HOME_DEFAULT = os.path.join(MODELS_DOWNLOAD_ROOT, "hf-cache")
os.environ.setdefault("HF_HOME", HF_HOME_DEFAULT)
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    os.path.join(os.environ["HF_HOME"], "hub"),
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    os.path.join(os.environ["HF_HOME"], "transformers"),
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DOWNLOAD_ROOT, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


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
    token = os.getenv("HF_TOKEN")
    return token or None
