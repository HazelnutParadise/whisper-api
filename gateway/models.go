package gateway

import "time"

type ModelInfo struct {
	ID         string  `json:"id"`
	Object     string  `json:"object"`
	OwnedBy    string  `json:"owned_by"`
	Permission []any   `json:"permission"`
	Created    int64   `json:"created"`
	Root       string  `json:"root"`
	Parent     *string `json:"parent"`
}

func supportedModels(now time.Time) []ModelInfo {
	created := now.Unix()
	return []ModelInfo{
		{
			ID:         "whisper-1",
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       "turbo",
		},
		{
			ID:         "turbo",
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       "turbo",
		},
		{
			ID:         "tts-1",
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       DefaultBackendTTSModel,
		},
		{
			ID:         "tts-1-hd",
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       DefaultBackendTTSModel,
		},
		{
			ID:         DefaultBackendTTSModel,
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       DefaultBackendTTSModel,
		},
		{
			ID:         LegacyBackendTTSModel,
			Object:     "model",
			OwnedBy:    "openai-compatible",
			Permission: []any{},
			Created:    created,
			Root:       DefaultBackendTTSModel,
		},
	}
}

func mapSpeechModel(model string) string {
	switch model {
	case "", "tts-1", "tts-1-hd", LegacyBackendTTSModel:
		return DefaultBackendTTSModel
	default:
		return model
	}
}
