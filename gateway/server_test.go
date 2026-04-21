package gateway

import (
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelsEndpointReturnsCombinedASRAndTTSModels(t *testing.T) {
	app := NewApp(Config{})

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("expected valid json response: %v", err)
	}

	want := map[string]bool{
		"whisper-1": false,
		"turbo":     false,
		"tts-1":     false,
		"tts-1-hd":  false,
		"coqui-tts": false,
	}
	for _, model := range payload.Data {
		if _, ok := want[model.ID]; ok {
			want[model.ID] = true
		}
	}

	for modelID, seen := range want {
		if !seen {
			t.Fatalf("expected model %q in response, got %+v", modelID, payload.Data)
		}
	}
}

func TestSpeechEndpointRejectsUnsupportedSSEStreamFormat(t *testing.T) {
	app := NewApp(Config{})

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/audio/speech",
		strings.NewReader(`{"model":"tts-1","input":"hello","voice":"alloy","stream":true,"stream_format":"sse"}`),
	)
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d with body %s", rec.Code, rec.Body.String())
	}

	if !strings.Contains(rec.Body.String(), "stream_format") {
		t.Fatalf("expected validation error mentioning stream_format, got %q", rec.Body.String())
	}
}

func TestOpenAPIEndpointIncludesSpeechPath(t *testing.T) {
	app := NewApp(Config{})

	req := httptest.NewRequest(http.MethodGet, "/openapi.json", nil)
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var payload struct {
		Paths map[string]any `json:"paths"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("expected valid openapi json: %v", err)
	}

	if _, ok := payload.Paths["/v1/audio/speech"]; !ok {
		t.Fatalf("expected /v1/audio/speech path in openapi spec, got %+v", payload.Paths)
	}
}

func TestSpeechEndpointMapsOpenAIAliasToCoquiBackendModel(t *testing.T) {
	tts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/speech" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("expected valid json body: %v", err)
		}
		if got := payload["model"]; got != DefaultBackendTTSModel {
			t.Fatalf("expected model %q, got %v", DefaultBackendTTSModel, got)
		}
		if got := payload["response_format"]; got != "pcm" {
			t.Fatalf("expected gateway to request pcm from backend, got %v", got)
		}

		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte{0x00, 0x01, 0x02, 0x03})
	}))
	defer tts.Close()

	app := NewApp(Config{TTSBaseURL: tts.URL})

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/audio/speech",
		strings.NewReader(`{"model":"tts-1","input":"hello","voice":"alloy","response_format":"pcm"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d with body %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "audio/pcm" {
		t.Fatalf("expected audio/pcm content type, got %q", got)
	}
	if body := rec.Body.Bytes(); string(body) != string([]byte{0x00, 0x01, 0x02, 0x03}) {
		t.Fatalf("unexpected audio body: %v", body)
	}
}

func TestSpeechEndpointMapsPublicCoquiModelToCurrentBackendModel(t *testing.T) {
	tts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("expected valid json body: %v", err)
		}
		if got := payload["model"]; got != DefaultBackendTTSModel {
			t.Fatalf("expected model %q, got %v", DefaultBackendTTSModel, got)
		}

		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte{0x00, 0x01})
	}))
	defer tts.Close()

	app := NewApp(Config{TTSBaseURL: tts.URL})

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/audio/speech",
		strings.NewReader(`{"model":"coqui-tts","input":"hello","voice":"alloy","response_format":"pcm"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d with body %s", rec.Code, rec.Body.String())
	}
}

func TestSpeechEndpointRejectsUpstreamRepositoryModelIDs(t *testing.T) {
	app := NewApp(Config{})

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/audio/speech",
		strings.NewReader(`{"model":"eustlb/higgs-audio-v2-generation-3B-base","input":"hello","voice":"alloy","response_format":"pcm"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d with body %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "coqui-tts") {
		t.Fatalf("expected validation error to mention public model id, got %q", rec.Body.String())
	}
}

func TestTranscriptionsEndpointProxiesMultipartBodyToASRBackend(t *testing.T) {
	asr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := r.ParseMultipartForm(1024 * 1024); err != nil {
			t.Fatalf("expected multipart form body: %v", err)
		}
		if got := r.FormValue("model"); got != "whisper-1" {
			t.Fatalf("expected model whisper-1, got %q", got)
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			t.Fatalf("expected file upload: %v", err)
		}
		defer file.Close()
		contents, err := io.ReadAll(file)
		if err != nil {
			t.Fatalf("expected readable upload body: %v", err)
		}
		if string(contents) != "audio" {
			t.Fatalf("unexpected upload contents: %q", string(contents))
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"ok"}`))
	}))
	defer asr.Close()

	body := &strings.Builder{}
	writer := multipart.NewWriter(body)
	if err := writer.WriteField("model", "whisper-1"); err != nil {
		t.Fatalf("expected model field write to succeed: %v", err)
	}
	part, err := writer.CreateFormFile("file", "sample.wav")
	if err != nil {
		t.Fatalf("expected file part creation to succeed: %v", err)
	}
	if _, err := io.Copy(part, strings.NewReader("audio")); err != nil {
		t.Fatalf("expected file write to succeed: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("expected multipart writer close to succeed: %v", err)
	}

	app := NewApp(Config{ASRBaseURL: asr.URL})
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader(body.String()))
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.ContentLength = int64(len(body.String()))
	rec := httptest.NewRecorder()

	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d with body %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("expected application/json content type, got %q", got)
	}
	if rec.Body.String() != `{"text":"ok"}` {
		t.Fatalf("unexpected proxy body: %q", rec.Body.String())
	}
}
