# AGENTS.md

## Follow-ups

- **Transcription still rides on one long-lived HTTP request.**
  `POST /v1/audio/transcriptions` holds a single connection open from upload until
  the ASR service finishes, so every timeout between the caller and this service
  becomes a hard ceiling on recording length. The gateway's own ten-minute limit is
  gone (see the comment in `gateway/config.go`), but the shape of the API has not
  changed, and a reverse proxy or browser in front of it can still cut a long run
  short.

  A job-style flow would decouple the two: the upload returns an id, the caller
  polls for the result. That means a second endpoint so the OpenAI-compatible path
  keeps working, plus matching changes in the `super-minutes` and `super-captions`
  frontends, which currently read an NDJSON keepalive stream from their own
  `/api/transcribe` route.

  Worth deciding only after measuring how long a real multi-hour recording takes
  end to end. If it now finishes without being cut off, this work is not needed.
