# whisper-api

An OpenAI-compatible speech stack with three services:

- `gateway`: Go entrypoint for `/v1/models`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/openapi.json`, `/docs`, and `/redoc`
- `asr`: FastAPI + WhisperX backend for transcription
- `coqui-tts`: FastAPI + Coqui TTS backend for text-to-speech

Only the Go gateway is exposed publicly. The ASR and TTS services stay on the
internal Docker network.

## Architecture

```text
client
  -> gateway (Go, port 5000)
       -> asr (FastAPI, WhisperX)
       -> coqui-tts (FastAPI, Coqui TTS)
```

## Endpoints

### `GET /v1/models`

Returns a combined model list for ASR and TTS:

- `whisper-1`
- `turbo`
- `tts-1`
- `tts-1-hd`
- `coqui-tts`

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

Supported public TTS model values:

- `tts-1`
- `tts-1-hd`
- `coqui-tts`

Supported `response_format` values:

- `mp3`
- `opus`
- `aac`
- `flac`
- `wav`
- `pcm`

`stream_format` currently supports only `audio`. `sse` is rejected with `400`.

The `voice` field is accepted for OpenAI compatibility. If the selected Coqui
model exposes built-in speaker names, the backend uses a matching `voice`;
otherwise it falls back to the first built-in speaker.

## Coqui TTS Notes

The TTS backend uses the Coqui Python API from `coqui-ai/TTS` instead of the
previous Higgs Audio implementation.

Default backend model:

```text
tts_models/multilingual/multi-dataset/xtts_v2
```

The public API does not expose backend repository/model names directly. The
gateway maps `tts-1`, `tts-1-hd`, and `coqui-tts` to the configured Coqui model.

The default model is XTTS v2, a multilingual Coqui model. The backend keeps the
deployment simple: language is fixed to `zh-cn`, the model loads lazily on first
request, and the TTS worker exits after each response so Docker releases its CUDA
context.

Coqui TTS package support is Python `>=3.9,<3.12`, so the Docker TTS service uses
the Python 3.11 PyTorch CUDA runtime image.

## Docker Compose

Create `.env` from `.env.example` only if you need a Hugging Face token:

```bash
HF_TOKEN=hf_xxx
```

Model caches are fixed inside the images and backed by the mounted volumes:

- ASR Hugging Face cache: `/app/models/hf-cache` via `/mnt/ssd1/whisper/models:/app/models`
- Coqui cache: `/root/.local/share/tts` via `/mnt/ssd1/whisper/coqui-cache:/root/.local/share/tts`
- TTS Hugging Face cache: `/root/.local/share/tts/hf-cache`, under the same Coqui cache mount

Start all three services:

```bash
docker network create infra-net
docker compose up --build
```

The Coqui backend exposes `/healthz`, and compose waits for that endpoint before
starting the public gateway. `/healthz` only checks that the HTTP service is
alive; the Coqui model is loaded lazily on the first TTS request.

The ASR backend unloads WhisperX after each request. The TTS backend unloads
Coqui and exits its worker after each response so Docker fully releases the TTS
CUDA context before a later STT request loads WhisperX on the same GPU.

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

Python TTS tests:

```bash
.venv\Scripts\python.exe -m unittest tests.test_tts_service -v
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

that usually means the `coqui-tts` container is not listening yet.

Check:

```bash
docker compose ps
docker compose logs coqui-tts
nvidia-smi
```

Common causes:

- the Coqui model is still downloading or loading into GPU memory
- ASR and TTS are both trying to load large models onto the same GPU
- the GPU still has orphaned model processes from previous containers
- the selected Coqui model requires a speaker but exposes no built-in speaker names
