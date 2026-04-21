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

The `voice` field is accepted for OpenAI compatibility. The default Coqui model
is a single-speaker model, so voice changes require switching to a multi-speaker
or voice-cloning Coqui model and setting `COQUI_TTS_SPEAKER`,
`COQUI_TTS_LANGUAGE`, or `COQUI_TTS_SPEAKER_WAV`.

## Coqui TTS Notes

The TTS backend uses the Coqui Python API from `coqui-ai/TTS` instead of the
previous Higgs Audio implementation.

Default backend model:

```text
tts_models/en/ljspeech/vits
```

The public API does not expose backend repository/model names directly. The
gateway maps `tts-1`, `tts-1-hd`, and `coqui-tts` to the configured Coqui model.

Coqui TTS package support is Python `>=3.9,<3.12`, so the Docker TTS service uses
the Python 3.11 PyTorch CUDA runtime image.

## Docker Compose

Create `.env` from `.env.example` and set at least:

```bash
HF_TOKEN=hf_xxx
HF_HUB_DISABLE_XET=1
CUDA_VISIBLE_DEVICES=0
ASR_CUDA_VISIBLE_DEVICES=0
COQUI_CUDA_VISIBLE_DEVICES=0
WHISPERX_UNLOAD_AFTER_REQUEST=1
ASR_MODELS_PATH=/mnt/ssd1/whisper/models
COQUI_CACHE_PATH=/mnt/ssd1/whisper/coqui-cache
COQUI_TTS_MODEL=tts_models/en/ljspeech/vits
COQUI_TTS_DEVICE=cuda
COQUI_TTS_PRELOAD_ON_STARTUP=0
COQUI_TTS_HEALTH_REQUIRE_MODEL=0
COQUI_TTS_UNLOAD_AFTER_REQUEST=1
COQUI_TTS_EXIT_AFTER_REQUEST=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Start all three services:

```bash
docker network create infra-net
docker compose up --build
```

The Coqui backend exposes `/healthz`, and compose waits for that endpoint before
starting the public gateway. By default `/healthz` only checks that the HTTP
service is alive; the Coqui model is loaded lazily on the first TTS request. Set
`COQUI_TTS_PRELOAD_ON_STARTUP=1` and `COQUI_TTS_HEALTH_REQUIRE_MODEL=1` only
when Docker health should mean "TTS model already loaded".

By default `WHISPERX_UNLOAD_AFTER_REQUEST=1` and
`COQUI_TTS_UNLOAD_AFTER_REQUEST=1` unload models and clear CUDA cache after each
STT/TTS request. `COQUI_TTS_EXIT_AFTER_REQUEST=1` also exits the TTS worker
after the HTTP response is sent so Docker can fully release the TTS CUDA context
before a later STT request loads WhisperX on the same GPU.

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
- the selected Coqui model requires a `speaker`, `language`, or `speaker_wav`

For a multi-GPU host, split ASR and TTS:

```bash
ASR_CUDA_VISIBLE_DEVICES=0
COQUI_CUDA_VISIBLE_DEVICES=1
```
