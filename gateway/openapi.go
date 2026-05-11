package gateway

func buildOpenAPISpec() map[string]any {
	return map[string]any{
		"openapi": "3.1.0",
		"info": map[string]any{
			"title":   "OpenAI-Compatible Speech Gateway",
			"version": "0.1.0",
		},
		"paths": map[string]any{
			"/healthz": map[string]any{
				"get": map[string]any{
					"summary": "Health check",
					"responses": map[string]any{
						"200": map[string]any{
							"description": "Gateway is healthy",
						},
					},
				},
			},
			"/v1/models": map[string]any{
				"get": map[string]any{
					"summary": "List supported ASR/TTS models",
					"responses": map[string]any{
						"200": map[string]any{
							"description": "Combined model list",
						},
					},
				},
			},
			"/v1/audio/transcriptions": map[string]any{
				"post": map[string]any{
					"summary": "Proxy transcription requests to the ASR backend",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"multipart/form-data": map[string]any{
								"schema": map[string]any{
									"type":     "object",
									"required": []string{"file"},
									"properties": map[string]any{
										"file": map[string]any{
											"type":        "string",
											"format":      "binary",
											"description": "Audio or video file to transcribe. Any format supported by ffmpeg can be uploaded.",
										},
										"model": map[string]any{
											"type":        "string",
											"enum":        []string{"whisper-1", "turbo"},
											"default":     "whisper-1",
											"description": "Transcription model. `whisper-1` is a compatibility alias for WhisperX `turbo`.",
										},
										"language": map[string]any{
											"type":        "string",
											"description": "Optional ISO language code (e.g. `en`, `zh`, `ja`). Auto-detected when omitted.",
										},
										"advanced": map[string]any{
											"type":        "boolean",
											"default":     false,
											"description": "When false, return only `{text}`. When true, return aligned timestamps, language, and optional diarization.",
										},
										"diarize": map[string]any{
											"type":        "boolean",
											"default":     true,
											"description": "Enable speaker diarization in advanced mode. Ignored when `advanced=false`. Requires `HF_TOKEN`.",
										},
										"min_speakers": map[string]any{
											"type":        "integer",
											"description": "Lower bound for number of speakers. Used only when `advanced=true` and `diarize=true`.",
										},
										"max_speakers": map[string]any{
											"type":        "integer",
											"description": "Upper bound for number of speakers. Used only when `advanced=true` and `diarize=true`.",
										},
									},
								},
							},
						},
					},
					"responses": map[string]any{
						"200": map[string]any{"description": "Transcription result"},
						"400": map[string]any{"description": "Validation error"},
						"503": map[string]any{"description": "ASR backend unavailable"},
					},
				},
			},
			"/v1/audio/speech": map[string]any{
				"post": map[string]any{
					"summary": "Generate speech with Coqui TTS through an OpenAI-compatible interface",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"application/json": map[string]any{
								"schema": map[string]any{
									"type":     "object",
									"required": []string{"model", "input", "voice"},
									"properties": map[string]any{
										"model": map[string]any{
											"type":        "string",
											"enum":        []string{"tts-1", "tts-1-hd", PublicBackendTTSModel},
											"description": "Public TTS model name. All values map to the configured Coqui backend model.",
										},
										"input": map[string]any{
											"type":        "string",
											"description": "Text to synthesize into speech.",
										},
										"voice": map[string]any{
											"type":        "string",
											"description": "Voice name. Kept for OpenAI compatibility; falls back to a built-in Coqui speaker if no matching name exists.",
										},
										"instructions": map[string]any{
											"type":        "string",
											"description": "Optional style/instruction hint forwarded to the backend. Ignored if the backend does not support it.",
										},
										"response_format": map[string]any{
											"type":        "string",
											"enum":        []string{"mp3", "opus", "aac", "flac", "wav", "pcm"},
											"default":     "mp3",
											"description": "Audio container/codec returned to the client. The gateway transcodes from the backend's PCM output.",
										},
										"speed": map[string]any{
											"type":        "number",
											"minimum":     0.25,
											"maximum":     4.0,
											"default":     1.0,
											"description": "Playback speed multiplier applied during transcoding.",
										},
										"stream": map[string]any{
											"type":        "boolean",
											"default":     false,
											"description": "If true, stream audio bytes back with chunked transfer encoding.",
										},
										"stream_format": map[string]any{
											"type":        "string",
											"enum":        []string{"audio"},
											"default":     "audio",
											"description": "Streaming format. Only `audio` is supported; `sse` is rejected with 400.",
										},
									},
								},
							},
						},
					},
					"responses": map[string]any{
						"200": map[string]any{"description": "Audio bytes"},
						"400": map[string]any{"description": "Validation error"},
						"503": map[string]any{"description": "TTS backend unavailable"},
					},
				},
			},
		},
	}
}
