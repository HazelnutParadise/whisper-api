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
					"responses": map[string]any{
						"200": map[string]any{"description": "Transcription result"},
					},
				},
			},
			"/v1/audio/speech": map[string]any{
				"post": map[string]any{
					"summary": "Generate speech with Higgs TTS through an OpenAI-compatible interface",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"application/json": map[string]any{
								"schema": map[string]any{
									"type":     "object",
									"required": []string{"model", "input", "voice"},
									"properties": map[string]any{
										"model": map[string]any{
											"type": "string",
											"enum": []string{"tts-1", "tts-1-hd", PublicBackendTTSModel},
										},
										"input":        map[string]any{"type": "string"},
										"voice":        map[string]any{"type": "string"},
										"instructions": map[string]any{"type": "string"},
										"response_format": map[string]any{
											"type": "string",
											"enum": []string{"mp3", "opus", "aac", "flac", "wav", "pcm"},
										},
										"speed": map[string]any{
											"type":    "number",
											"minimum": 0.25,
											"maximum": 4.0,
										},
										"stream": map[string]any{"type": "boolean"},
										"stream_format": map[string]any{
											"type": "string",
											"enum": []string{"audio"},
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
