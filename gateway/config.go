package gateway

import (
	"net/http"
	"os"
	"time"
)

const (
	DefaultSamplingRate    = 24000
	DefaultBackendTTSModel = "tts_models/en/ljspeech/vits"
	PublicBackendTTSModel  = "coqui-tts"
)

type Config struct {
	ASRBaseURL string
	TTSBaseURL string
	HTTPClient *http.Client
}

func NewConfigFromEnv() Config {
	return Config{
		ASRBaseURL: os.Getenv("ASR_BASE_URL"),
		TTSBaseURL: os.Getenv("TTS_BASE_URL"),
		HTTPClient: &http.Client{Timeout: 10 * time.Minute},
	}
}

func (c Config) client() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 10 * time.Minute}
}
