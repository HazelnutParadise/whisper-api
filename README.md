# whisper-api

An OpenAI-compatible speech stack with three services:

- `gateway`: Go entrypoint for `/v1/models`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/openapi.json`, `/docs`, and `/redoc`
- `asr`: FastAPI + WhisperX backend for transcription
- `higgs-tts`: Higgs Audio vLLM backend for TTS with CUDA

Only the Go gateway is exposed publicly. The ASR and TTS services stay on the
internal Docker network.

## Architecture

```text
client
  -> gateway (Go, port 5000)
       -> asr (FastAPI, WhisperX)
       -> higgs-tts (bosonai/higgs-audio-vllm)
```

## Endpoints

### `GET /v1/models`

Returns a combined model list for ASR and TTS:

- `whisper-1`
- `turbo`
- `tts-1`
- `tts-1-hd`
- `bosonai/higgs-audio-v2-generation-3B-base`

### `POST /v1/audio/transcriptions`

OpenAI-style multipart transcription endpoint. The gateway proxies the request
to the ASR service.

Form fields:

- `file`
- `model`
- `language`
- `advanced`
- `diarize`
- `min_speakers`
- `max_speakers`

### `POST /v1/audio/speech`

OpenAI-style JSON TTS endpoint. The gateway accepts:

- `model`
- `input`
- `voice`
- `instructions`
- `response_format`
- `speed`
- `stream`
- `stream_format`

Supported `response_format` values:

- `mp3`
- `opus`
- `aac`
- `flac`
- `wav`
- `pcm`

`stream_format` currently supports only `audio`. `sse` is rejected with `400`.

## Higgs Audio Notes

The compose file currently uses the official Higgs vLLM image:

- `bosonai/higgs-audio-vllm:latest`

This image is referenced in the upstream repo's vLLM example:

- https://github.com/boson-ai/higgs-audio/tree/main/examples/vllm

Higgs Audio is multilingual, but it should not be treated as "all languages
supported." The strongest evidence currently points to English, Chinese
(primarily Mandarin), Korean, German, and Spanish. Other languages should be
treated as best-effort.

## Latest-First Version Policy

This repo follows a latest-first policy:

- prefer current stable Go for the gateway
- prefer current compatible Python packages for the ASR service
- use the current upstream Higgs vLLM image first
- only pin or roll back further when build or runtime verification fails

The local verified Python test environment in this repo is currently the
existing `.venv` with Python `3.12.11`.

## Local Python ASR Development

Install dependencies in the existing virtual environment:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the ASR app directly:

```bash
.venv\Scripts\python.exe app.py
```

Or with uvicorn:

```bash
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 5000
```

## Docker Compose

Create `.env` from `.env.example` and set at least:

```bash
HF_TOKEN=hf_xxx
CUDA_VISIBLE_DEVICES=0
```

Start all three services:

```bash
docker compose up --build
```

Public gateway endpoint:

- `http://localhost:5148`

Docs:

- `http://localhost:5148/docs`
- `http://localhost:5148/redoc`
- `http://localhost:5148/openapi.json`

## Examples

### Transcription

```bash
curl -X POST "http://localhost:5148/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### Advanced transcription

```bash
curl -X POST "http://localhost:5148/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "advanced=true" \
  -F "diarize=true" \
  -F "min_speakers=1" \
  -F "max_speakers=3"
```

### TTS

```bash
curl -X POST "http://localhost:5148/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "en_woman",
    "input": "Today is a wonderful day to build something people love.",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Testing

Python ASR tests:

```bash
.venv\Scripts\python.exe -m unittest tests.test_transcriptions_endpoint -v
```

Go gateway tests:

```bash
go test ./gateway/...
```
