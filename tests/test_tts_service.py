import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.testclient import TestClient

import tts_service.app as tts_app
from tts_service.app import (
    DEFAULT_BACKEND_TTS_MODEL,
    DEFAULT_COQUI_LANGUAGE,
    EngineBundle,
    PUBLIC_BACKEND_TTS_MODEL,
    app,
    cleanup_request_runtime_refs,
    coqui_tts_kwargs,
)


class FakeCoquiModel:
    def __init__(self, pcm_bytes=b"\x00\x01\x02\x03"):
        self.calls = []
        self.is_multi_speaker = False
        self.is_multi_lingual = False
        self.speakers = []
        self.languages = []
        self.pcm_bytes = pcm_bytes

    def tts_to_file(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["file_path"]).write_bytes(b"fake-wav")


class TTSServiceTests(unittest.TestCase):
    def test_tts_runtime_dependencies_are_declared(self):
        requirements = Path("tts_service/requirements.txt").read_text()
        self.assertIn("TTS", requirements)
        self.assertIn("soundfile", requirements)
        self.assertIn("transformers==4.44.2", requirements)

    def test_speech_endpoint_returns_pcm_audio_from_coqui_engine(self):
        fake_model = FakeCoquiModel()
        fake_engine = EngineBundle(model=fake_model)

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
            patch("tts_service.app.wav_file_to_pcm16le", return_value=b"\x00\x01"),
            patch("tts_service.app.unload_engine", autospec=True) as unload_engine,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": PUBLIC_BACKEND_TTS_MODEL,
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/pcm")
        self.assertEqual(response.content, b"\x00\x01")
        self.assertEqual(fake_model.calls[0]["text"], "hello world")
        unload_engine.assert_called_once()

    def test_speech_endpoint_accepts_internal_backend_model_from_gateway(self):
        fake_engine = EngineBundle(model=FakeCoquiModel())

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
            patch("tts_service.app.wav_file_to_pcm16le", return_value=b"\x00\x01"),
            patch("tts_service.app.unload_engine", autospec=True),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": DEFAULT_BACKEND_TTS_MODEL,
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 200)

    def test_speech_endpoint_rejects_upstream_higgs_model_ids(self):
        with patch("tts_service.app.preload_engine_async", autospec=True):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "eustlb/higgs-audio-v2-generation-3B-base",
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn(PUBLIC_BACKEND_TTS_MODEL, response.json()["detail"])

    def test_cleanup_request_runtime_refs_drops_per_request_references(self):
        runtime_refs = {"bundle": object(), "inputs": object()}

        with patch("tts_service.app.clear_cuda_cache", autospec=True) as clear_cuda_cache:
            cleanup_request_runtime_refs(runtime_refs)

        self.assertEqual(runtime_refs, {})
        clear_cuda_cache.assert_called_once()

    def test_speech_endpoint_exits_after_response_to_release_cuda_context(self):
        fake_engine = EngineBundle(model=FakeCoquiModel())

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
            patch("tts_service.app.wav_file_to_pcm16le", return_value=b"\x00\x01"),
            patch("tts_service.app.unload_engine", autospec=True),
            patch("tts_service.app.should_exit_after_response", return_value=True),
            patch("tts_service.app.os._exit", autospec=True) as exit_process,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": PUBLIC_BACKEND_TTS_MODEL,
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 200)
        exit_process.assert_called_once_with(0)

    def test_speech_endpoint_reports_cuda_oom_as_service_unavailable(self):
        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch(
                "tts_service.app.get_engine",
                side_effect=RuntimeError("CUDA out of memory. Tried to allocate 48.00 MiB."),
            ),
            patch("tts_service.app.unload_engine", autospec=True) as unload_engine,
            patch("tts_service.app.clear_cuda_cache", autospec=True) as clear_cuda_cache,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": PUBLIC_BACKEND_TTS_MODEL,
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GPU memory", response.json()["detail"])
        unload_engine.assert_called_once()
        self.assertGreaterEqual(clear_cuda_cache.call_count, 1)

    def test_coqui_kwargs_uses_default_language_and_matching_builtin_speaker(self):
        payload = Mock()
        payload.voice = "speaker-2"
        model = FakeCoquiModel()
        model.is_multi_speaker = True
        model.speakers = ["speaker-1", "speaker-2"]

        kwargs = coqui_tts_kwargs(payload, model)

        self.assertEqual(kwargs["language"], DEFAULT_COQUI_LANGUAGE)
        self.assertEqual(kwargs["speaker"], "speaker-2")

    def test_coqui_kwargs_falls_back_to_first_builtin_speaker(self):
        payload = Mock()
        payload.voice = "alloy"
        model = FakeCoquiModel()
        model.is_multi_speaker = True
        model.speakers = ["speaker-1", "speaker-2"]

        kwargs = coqui_tts_kwargs(payload, model)

        self.assertEqual(kwargs["speaker"], "speaker-1")

    def test_coqui_kwargs_defaults_to_multilingual_language(self):
        payload = Mock()
        model = FakeCoquiModel()
        model.is_multi_lingual = True

        kwargs = coqui_tts_kwargs(payload, model)

        self.assertEqual(kwargs["language"], DEFAULT_COQUI_LANGUAGE)

    def test_coqui_kwargs_requires_speaker_for_multispeaker_models(self):
        payload = Mock()
        model = FakeCoquiModel()
        model.is_multi_speaker = True

        with self.assertRaisesRegex(Exception, "requires a speaker"):
            coqui_tts_kwargs(payload, model)

    def test_healthz_reports_loading_without_blocking(self):
        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch.object(tts_app, "_ENGINE", None),
            patch.object(tts_app, "_ENGINE_LOADING", True, create=True),
            patch.object(tts_app, "_ENGINE_ERROR", None, create=True),
            patch(
                "tts_service.app.get_engine",
                side_effect=AssertionError("healthz should not block on get_engine"),
            ),
        ):
            with TestClient(app) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["engine"], "loading")

    def test_healthz_reports_failed_without_failing_container_health(self):
        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch.object(tts_app, "_ENGINE", None),
            patch.object(tts_app, "_ENGINE_LOADING", False, create=True),
            patch.object(
                tts_app,
                "_ENGINE_ERROR",
                RuntimeError("boom"),
                create=True,
            ),
            patch(
                "tts_service.app.get_engine",
                side_effect=AssertionError("healthz should report cached preload errors"),
            ),
        ):
            with TestClient(app) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["engine"], "failed")

    def test_healthz_is_ok_without_loaded_engine_by_default(self):
        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch.object(tts_app, "_ENGINE", None),
            patch.object(tts_app, "_ENGINE_LOADING", False, create=True),
            patch.object(tts_app, "_ENGINE_ERROR", None, create=True),
        ):
            with TestClient(app) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["engine"], "not_loaded")

    def test_startup_preload_is_disabled_by_default(self):
        with (
            patch("tts_service.app.preload_engine_async", Mock()) as preload,
        ):
            tts_app.startup_preload()

        preload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
