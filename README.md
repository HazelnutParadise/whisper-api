# whisper-api

A FastAPI-based Whisper transcription service compatible with OpenAI's API.

## Installation

```bash
pip install -r requirements.txt
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

### POST /v1/audio/transcriptions

Transcribe audio file to text.

**Request:**

- `file`: Audio file (multipart/form-data)
- `model`: Model name (optional, default: "whisper-1")

**Response:**

```json
{
  "text": "Transcription text here..."
}
```

## Example

Using curl:

```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav" \
     -F "model=whisper-1"
```
