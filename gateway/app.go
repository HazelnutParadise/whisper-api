package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type App struct {
	cfg    Config
	router *http.ServeMux
}

type errorResponse struct {
	Error  string `json:"error,omitempty"`
	Detail string `json:"detail,omitempty"`
}

type speechRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	Instructions   string  `json:"instructions,omitempty"`
	ResponseFormat string  `json:"response_format,omitempty"`
	Speed          float64 `json:"speed,omitempty"`
	Stream         bool    `json:"stream,omitempty"`
	StreamFormat   string  `json:"stream_format,omitempty"`
}

func NewApp(cfg Config) http.Handler {
	app := &App{
		cfg:    cfg,
		router: http.NewServeMux(),
	}

	app.router.HandleFunc("GET /healthz", app.handleHealthz)
	app.router.HandleFunc("GET /v1/models", app.handleModels)
	app.router.HandleFunc("POST /v1/audio/transcriptions", app.handleTranscriptions)
	app.router.HandleFunc("POST /v1/audio/speech", app.handleSpeech)
	app.router.HandleFunc("GET /openapi.json", app.handleOpenAPI)
	app.router.HandleFunc("GET /docs", app.handleDocs)
	app.router.HandleFunc("GET /redoc", app.handleRedoc)
	return app.router
}

func (a *App) writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func (a *App) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	a.writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (a *App) handleModels(w http.ResponseWriter, _ *http.Request) {
	a.writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   supportedModels(time.Now()),
	})
}

func (a *App) handleOpenAPI(w http.ResponseWriter, _ *http.Request) {
	a.writeJSON(w, http.StatusOK, buildOpenAPISpec())
}

func (a *App) handleDocs(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, `<!doctype html>
<html>
<head>
  <title>Speech Gateway Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({ url: '/openapi.json', dom_id: '#swagger-ui' });
  </script>
</body>
</html>`)
}

func (a *App) handleRedoc(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, `<!doctype html>
<html>
<head><title>Speech Gateway ReDoc</title></head>
<body>
  <redoc spec-url="/openapi.json"></redoc>
  <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
</body>
</html>`)
}

func (a *App) handleTranscriptions(w http.ResponseWriter, r *http.Request) {
	if a.cfg.ASRBaseURL == "" {
		a.writeJSON(w, http.StatusServiceUnavailable, errorResponse{Detail: "ASR backend is not configured."})
		return
	}

	baseURL, err := url.Parse(a.cfg.ASRBaseURL)
	if err != nil {
		a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: "ASR base URL is invalid."})
		return
	}

	target := baseURL.ResolveReference(&url.URL{Path: "/v1/audio/transcriptions"})
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target.String(), r.Body)
	if err != nil {
		a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: "Failed to create ASR request."})
		return
	}
	req.Header = r.Header.Clone()
	resp, err := a.cfg.client().Do(req)
	if err != nil {
		a.writeJSON(w, http.StatusBadGateway, errorResponse{Detail: fmt.Sprintf("ASR backend request failed: %v", err)})
		return
	}
	defer resp.Body.Close()

	copyResponse(w, resp)
}

func (a *App) handleSpeech(w http.ResponseWriter, r *http.Request) {
	var payload speechRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		a.writeJSON(w, http.StatusBadRequest, errorResponse{Detail: "Invalid JSON body."})
		return
	}

	if payload.Model == "" || payload.Input == "" || payload.Voice == "" {
		a.writeJSON(w, http.StatusBadRequest, errorResponse{Detail: "model, input, and voice are required."})
		return
	}
	if !isSupportedSpeechModel(payload.Model) {
		a.writeJSON(
			w,
			http.StatusBadRequest,
			errorResponse{
				Detail: "model must be one of tts-1, tts-1-hd, higgs-audio-v2-generation-3b.",
			},
		)
		return
	}

	if payload.ResponseFormat == "" {
		payload.ResponseFormat = "mp3"
	}
	if payload.Speed == 0 {
		payload.Speed = 1
	}
	if payload.Speed < 0.25 || payload.Speed > 4.0 {
		a.writeJSON(w, http.StatusBadRequest, errorResponse{Detail: "speed must be between 0.25 and 4.0."})
		return
	}
	if payload.StreamFormat == "" {
		payload.StreamFormat = "audio"
	}
	if payload.StreamFormat != "audio" {
		a.writeJSON(w, http.StatusBadRequest, errorResponse{Detail: "stream_format only supports audio."})
		return
	}
	if !isSupportedResponseFormat(payload.ResponseFormat) {
		a.writeJSON(w, http.StatusBadRequest, errorResponse{Detail: "response_format must be one of mp3, opus, aac, flac, wav, pcm."})
		return
	}

	if a.cfg.TTSBaseURL == "" {
		a.writeJSON(w, http.StatusServiceUnavailable, errorResponse{Detail: "TTS backend is not configured."})
		return
	}

	upstreamPayload := map[string]any{
		"model":           mapSpeechModel(payload.Model),
		"voice":           payload.Voice,
		"input":           payload.Input,
		"response_format": "pcm",
	}
	if payload.Instructions != "" {
		upstreamPayload["instructions"] = payload.Instructions
	}
	if payload.Stream {
		upstreamPayload["stream"] = true
		upstreamPayload["stream_format"] = "audio"
	}

	body, err := json.Marshal(upstreamPayload)
	if err != nil {
		a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: "Failed to encode TTS request."})
		return
	}

	baseURL, err := url.Parse(a.cfg.TTSBaseURL)
	if err != nil {
		a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: "TTS base URL is invalid."})
		return
	}

	target := baseURL.ResolveReference(&url.URL{Path: "/v1/audio/speech"})
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target.String(), bytes.NewReader(body))
	if err != nil {
		a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: "Failed to create TTS request."})
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.cfg.client().Do(req)
	if err != nil {
		a.writeJSON(w, http.StatusBadGateway, errorResponse{Detail: fmt.Sprintf("TTS backend request failed: %v", err)})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		copyResponse(w, resp)
		return
	}

	w.Header().Set("Content-Type", contentTypeForFormat(payload.ResponseFormat))
	if payload.Stream {
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		if payload.ResponseFormat == "pcm" && payload.Speed == 1 {
			_, _ = io.Copy(w, resp.Body)
			return
		}
		if err := streamTranscodePCM(resp.Body, w, payload.ResponseFormat, payload.Speed); err != nil {
			// At this point headers are already sent; terminate the connection after best-effort body write.
			return
		}
		return
	}

	rawPCM, err := io.ReadAll(resp.Body)
	if err != nil {
		a.writeJSON(w, http.StatusBadGateway, errorResponse{Detail: fmt.Sprintf("Failed reading TTS audio: %v", err)})
		return
	}

	output := rawPCM
	if payload.ResponseFormat != "pcm" || payload.Speed != 1 {
		output, err = transcodePCM(rawPCM, payload.ResponseFormat, payload.Speed)
		if err != nil {
			a.writeJSON(w, http.StatusInternalServerError, errorResponse{Detail: err.Error()})
			return
		}
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(output)
}

func copyResponse(w http.ResponseWriter, resp *http.Response) {
	for key, values := range resp.Header {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func isSupportedResponseFormat(format string) bool {
	switch format {
	case "mp3", "opus", "aac", "flac", "wav", "pcm":
		return true
	default:
		return false
	}
}

func isSupportedSpeechModel(model string) bool {
	switch model {
	case "tts-1", "tts-1-hd", PublicBackendTTSModel:
		return true
	default:
		return false
	}
}

func Serve(ctx context.Context, cfg Config, addr string) error {
	srv := &http.Server{
		Addr:    addr,
		Handler: NewApp(cfg),
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
}
