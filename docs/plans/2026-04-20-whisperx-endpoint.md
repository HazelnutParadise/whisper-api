# WhisperX Endpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new FastAPI endpoint that uses WhisperX to return transcription text, aligned timestamps, and speaker diarization.

**Architecture:** Keep the existing Whisper endpoint unchanged and add a separate WhisperX endpoint. Load the base Whisper and WhisperX ASR models during startup, cache alignment and diarization resources lazily, and expose a response that includes transcript text, language, segments, word timestamps, and speaker labels.

**Tech Stack:** FastAPI, openai-whisper, whisperx, torch, unittest, unittest.mock

---

### Task 1: Add failing tests for the new endpoint

**Files:**
- Create: `tests/test_whisperx_endpoint.py`

**Step 1: Write the failing test**

Add tests that:
- call `POST /v1/audio/transcriptions/whisperx`
- verify a successful mocked WhisperX flow returns `text`, `language`, `segments`, and diarization metadata
- verify the endpoint returns `503` when diarization is requested but no Hugging Face token is configured

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_whisperx_endpoint -v`

Expected: FAIL because the endpoint and supporting helpers do not exist yet.

### Task 2: Implement WhisperX model loading and endpoint

**Files:**
- Modify: `app.py`

**Step 1: Add configuration and helper functions**

Add helpers for:
- startup model loading
- temporary file creation and cleanup
- lazy alignment model caching by language
- lazy diarization pipeline loading from env token

**Step 2: Add the new endpoint**

Implement `POST /v1/audio/transcriptions/whisperx` with multipart upload handling and optional form inputs for language, min speakers, and max speakers.

**Step 3: Run tests to verify they pass**

Run: `python -m unittest tests.test_whisperx_endpoint -v`

Expected: PASS

### Task 3: Update dependencies and docs

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `Dockerfile`

**Step 1: Add required dependencies**

Add `whisperx` and any missing runtime dependency needed by multipart uploads in local installs.

**Step 2: Document the endpoint**

Describe:
- new endpoint path
- response shape
- required diarization token env var
- core WhisperX env vars

**Step 3: Verify with focused tests**

Run: `python -m unittest tests.test_whisperx_endpoint -v`

Expected: PASS
