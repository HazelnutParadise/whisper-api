# whisper-api

A FastAPI transcription service that keeps the legacy OpenAI-style endpoint
`/v1/audio/transcriptions`, but runs WhisperX underneath.

## Installation

```bash
pip install -r requirements.txt
```

## Required Runtime Dependencies

- `ffmpeg` must be available on `PATH`
- WhisperX speaker diarization requires a Hugging Face read token for the
  `pyannote/speaker-diarization-community-1` model

WhisperX still uses the external `ffmpeg` CLI for audio decoding, so the
runtime image must include `ffmpeg`.

Example environment variables:

```bash
export WHISPERX_HF_TOKEN=hf_xxx
export WHISPERX_DEVICE=cuda
export WHISPERX_COMPUTE_TYPE=float16
export WHISPERX_BATCH_SIZE=16
```

For CPU-only execution:

```bash
export WHISPERX_DEVICE=cpu
export WHISPERX_COMPUTE_TYPE=int8
```

## Usage

Start the server:

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

## API

### `GET /v1/models`

Returns the available model names:

- `whisper-1`
- `turbo`

`whisper-1` is treated as an alias of WhisperX `turbo`.

### `POST /v1/audio/transcriptions`

Single transcription endpoint.

Form fields:

- `file`: audio file
- `model`: optional, defaults to `whisper-1`
- `language`: optional language code such as `en`, `zh`, `ja`
- `advanced`: optional boolean, defaults to `false`
- `diarize`: optional boolean, defaults to `true`
- `min_speakers`: optional integer
- `max_speakers`: optional integer

If a required WhisperX asset has not been downloaded yet, the service starts a
background download and returns `202 Accepted`. Retry the same request shortly.

## Behavior

### Simple mode

If `advanced=false`:

- WhisperX is still used underneath
- response is trimmed to plain text only
- diarization is skipped even if `diarize=true`

Response:

```json
{
  "text": "Transcription text here..."
}
```

If the ASR model is still downloading:

```json
{
  "status": "model_downloading",
  "detail": "Requested model assets are downloading. Retry shortly.",
  "resources": ["asr:turbo"]
}
```

### Advanced mode

If `advanced=true`:

- full WhisperX response is returned
- if `diarize=true`, speaker diarization is attempted
- diarization requires `WHISPERX_HF_TOKEN` or `HF_TOKEN`
- if `language` is omitted, the first successful ASR pass may still return
  `202 Accepted` while the detected language's alignment model downloads

Response:

```json
{
  "text": "hello world",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 1.2,
      "text": "hello world",
      "speaker": "SPEAKER_00",
      "words": [
        {
          "word": "hello",
          "start": 0.0,
          "end": 0.5,
          "speaker": "SPEAKER_00"
        }
      ]
    }
  ],
  "diarization": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 1.2
    }
  ],
  "speakers": ["SPEAKER_00"]
}
```

## Examples

Simple transcription:

```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

Equivalent explicit turbo request:

```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=turbo"
```

Advanced WhisperX output:

```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "advanced=true" \
  -F "diarize=true" \
  -F "min_speakers=1" \
  -F "max_speakers=3"
```
