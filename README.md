# whisper-api

An OpenAI-compatible speech stack with three services:

- `gateway`: Go entrypoint for `/v1/models`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/openapi.json`, `/docs`, and `/redoc`
- `asr`: FastAPI + WhisperX backend for transcription
- `higgs-tts`: native Python Higgs Audio backend for TTS with CUDA

Only the Go gateway is exposed publicly. The ASR and TTS services stay on the
internal Docker network.

## Architecture

```text
client
  -> gateway (Go, port 5000)
       -> asr (FastAPI, WhisperX)
       -> higgs-tts (native HiggsAudioServeEngine)
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

The native TTS backend accepts OpenAI voice aliases such as `alloy`, `ash`,
and `shimmer`, and maps them onto bundled Higgs prompt voices. It also accepts
direct Higgs prompt names such as `belinda`, `en_woman`, `en_man`,
`broom_salesman`, `mabel`, and `chadwick`.

## Higgs Audio Notes

The compose file builds a native Python Higgs service from the upstream repo and
uses `HiggsAudioServeEngine` directly instead of the published vLLM image.

This avoids the tokenizer and template incompatibilities we observed in
`bosonai/higgs-audio-vllm:latest`, while keeping the public API surface
OpenAI-compatible through the Go gateway.

Higgs Audio is multilingual, but it should not be treated as "all languages
supported." The strongest evidence currently points to English, Chinese
(primarily Mandarin), Korean, German, and Spanish. Other languages should be
treated as best-effort.

## Latest-First Version Policy

This repo follows a latest-first policy:

- prefer current stable Go for the gateway
- prefer current compatible Python packages for the ASR service
- use the current upstream native Higgs Python engine first
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
HF_HUB_DISABLE_XET=1
CUDA_VISIBLE_DEVICES=0
ASR_MODELS_PATH=/mnt/ssd1/whisper/models
HIGGS_CACHE_PATH=/mnt/ssd1/whisper/higgs-cache
```

Start all three services:

```bash
docker network create infra-net
docker compose up --build
```

The native Higgs backend exposes `/healthz`, and compose waits for that
endpoint before starting the public gateway.

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
    "voice": "alloy",
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

## TTS Troubleshooting

If `POST /v1/audio/speech` returns:

```text
502 Bad Gateway
TTS backend request failed: ... connect: connection refused
```

that usually means the `higgs-tts` container is not listening yet, not that the
OpenAI-compatible route shape is wrong.

Check:

```bash
docker compose ps
docker compose logs higgs-tts
```

Common causes:

- the native Higgs model is still downloading or loading into GPU memory
- the Hugging Face `hf-xet` download path is failing TLS handshakes in the container
- the container exited before the native `/healthz` endpoint became ready

The Higgs Audio model and tokenizer pages are public on Hugging Face, so this
failure is usually not caused by gated-model approval. Still, setting `HF_TOKEN`
is recommended for reliable Hub downloads and rate limiting.

If logs show `xet-core` retries with `tls handshake eof`, set:

```bash
HF_HUB_DISABLE_XET=1
```

This repo now defaults to that value in compose so model downloads fall back to
the regular Hub path instead of the Xet transfer client.
