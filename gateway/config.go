package gateway

import (
	"net/http"
	"os"
)

const (
	DefaultSamplingRate    = 24000
	DefaultBackendTTSModel = "tts_models/multilingual/multi-dataset/xtts_v2"
	PublicBackendTTSModel  = "coqui-tts"
)

type Config struct {
	ASRBaseURL string
	TTSBaseURL string
	HTTPClient *http.Client
}

// No Timeout is set on these clients on purpose. http.Client.Timeout covers the
// whole request, and a transcription can legitimately run for hours, so any
// fixed value here eventually cuts off a long recording mid-run. The backend
// requests carry the caller's context instead, so a client that goes away still
// tears the upstream request down.
func NewConfigFromEnv() Config {
	return Config{
		ASRBaseURL: os.Getenv("ASR_BASE_URL"),
		TTSBaseURL: os.Getenv("TTS_BASE_URL"),
		HTTPClient: &http.Client{},
	}
}

func (c Config) client() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}
