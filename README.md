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
export WHISPERX_AUTO_CHUNK_MIN_MB=100
export WHISPERX_CHUNK_SECONDS=600
export WHISPERX_CHUNK_OVERLAP_SECONDS=3
export WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM=0.7
export HF_HOME=./models/hf-cache
```

If you run the service with Docker Compose, create a local `.env` file in the
project root and set:

```bash
WHISPERX_HF_TOKEN=hf_xxx
```

The provided `docker-compose.yml` forwards this variable into the container.

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
model loading in the same request. The first request can be slower while model
assets are prepared.

If another request arrives while the same model is still loading, the service
waits for the in-flight load and reuses that result instead of starting another
download/load for the same asset.

When running with Docker Compose, Hugging Face caches are persisted under
`/app/models/hf-cache` through the existing `/app/models` volume.

## Automatic Chunked Transcription

For large uploads, the service can automatically split audio into chunks,
transcribe each chunk, and merge results.

- `WHISPERX_AUTO_CHUNK_MIN_MB`: file-size threshold in MB for auto chunking
  - default: `100`
  - set `0` or negative to disable auto chunking
- `WHISPERX_CHUNK_SECONDS`: chunk length in seconds
  - default: `600`
- `WHISPERX_CHUNK_OVERLAP_SECONDS`: overlap window in seconds
  - default: `3`
- `WHISPERX_CHUNK_SPEAKER_MERGE_MIN_SIM`: minimum similarity needed to merge
  chunk-local speaker labels into existing global speakers
  - range: `0` to `1`
  - default: `0.7`

When `advanced=true` and chunking is used, speaker labels are reconciled across
chunk boundaries so segment-local labels are normalized to global speaker IDs.
Non-chunked requests keep the original single-pass behavior.

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

### Advanced mode

If `advanced=true`:

- full WhisperX response is returned
- if `diarize=true`, speaker diarization is attempted
- diarization requires `WHISPERX_HF_TOKEN` or `HF_TOKEN`
- if `language` is omitted, the first request may take longer because the
  detected language alignment model is loaded before returning

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
