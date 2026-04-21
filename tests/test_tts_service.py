import unittest
import wave
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

import tts_service.app as tts_app
from tts_service.app import (
    DEFAULT_BACKEND_TTS_MODEL,
    DEFAULT_SAMPLING_RATE,
    EngineBundle,
    OPENAI_VOICE_ALIASES,
    PUBLIC_BACKEND_TTS_MODEL,
    app,
    prepare_outputs_for_decode,
)


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeModel:
    def __init__(self, response):
        self.device = "cpu"
        self.response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeProcessor:
    def __init__(self, decoded):
        self.decoded = decoded
        self.apply_calls = []
        self.decode_calls = []

    def apply_chat_template(self, *args, **kwargs):
        self.apply_calls.append((args, kwargs))
        return FakeInputs({"input_ids": "fake-input-ids"})

    def batch_decode(self, outputs):
        self.decode_calls.append(outputs)
        return self.decoded

    def save_audio(self, _decoded, output_path):
        pcm_samples = np.array([0, 16384, -16384], dtype=np.int16)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(DEFAULT_SAMPLING_RATE)
            wav_file.writeframes(pcm_samples.tobytes())


class FakeAudioTokenizer:
    def __init__(self, tensor):
        self.tensor = tensor

    def parameters(self):
        return iter([self.tensor])


class FakeGeneratedOutputs:
    def __init__(self):
        self.target_device = None

    def to(self, target_device):
        self.target_device = target_device
        return self


class TTSServiceTests(unittest.TestCase):
    def test_openai_voice_aliases_map_to_supported_voice_prompts(self):
        self.assertEqual(OPENAI_VOICE_ALIASES["alloy"], "belinda")
        self.assertEqual(OPENAI_VOICE_ALIASES["ash"], "en_man")
        self.assertEqual(OPENAI_VOICE_ALIASES["shimmer"], "en_woman")

    def test_speech_endpoint_returns_pcm_audio_from_native_engine(self):
        fake_outputs = {"audio_tokens": [1, 2, 3]}
        fake_model = FakeModel(fake_outputs)
        fake_processor = FakeProcessor(decoded=["decoded-audio"])
        fake_engine = EngineBundle(model=fake_model, processor=fake_processor)

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
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
        self.assertEqual(response.content, np.array([0, 16384, -16384], dtype=np.int16).tobytes())
        self.assertEqual(fake_processor.apply_calls[0][0][0][0]["role"], "system")
        self.assertEqual(fake_model.calls[0]["input_ids"], "fake-input-ids")
        self.assertEqual(fake_processor.decode_calls, [fake_outputs])
        unload_engine.assert_called_once()

    def test_speech_endpoint_can_keep_engine_loaded_when_configured(self):
        fake_outputs = {"audio_tokens": [1, 2, 3]}
        fake_engine = EngineBundle(
            model=FakeModel(fake_outputs),
            processor=FakeProcessor(decoded=["decoded-audio"]),
        )

        with (
            patch.dict("os.environ", {"HIGGS_UNLOAD_AFTER_REQUEST": "0"}),
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
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
        unload_engine.assert_not_called()

    def test_prepare_outputs_for_decode_moves_tensor_to_audio_tokenizer_device(self):
        processor = Mock()
        processor.audio_tokenizer = FakeAudioTokenizer(torch.empty(1, device="cpu"))
        outputs = FakeGeneratedOutputs()

        prepared = prepare_outputs_for_decode(outputs, processor)

        self.assertIs(prepared, outputs)
        self.assertEqual(outputs.target_device.type, "cpu")

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
        clear_cuda_cache.assert_called_once()

    def test_speech_endpoint_reports_generation_errors_as_json(self):
        fake_engine = EngineBundle(
            model=FakeModel(response=None),
            processor=FakeProcessor(decoded=["decoded-audio"]),
        )
        fake_engine.model.generate = Mock(side_effect=RuntimeError("decode failed"))

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
            patch("tts_service.app.unload_engine", autospec=True),
            patch("tts_service.app.clear_cuda_cache", autospec=True),
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

        self.assertEqual(response.status_code, 500)
        self.assertIn("decode failed", response.json()["detail"])

    def test_speech_endpoint_accepts_internal_backend_model_from_gateway(self):
        fake_outputs = {"audio_tokens": [1, 2, 3]}
        fake_engine = EngineBundle(
            model=FakeModel(fake_outputs),
            processor=FakeProcessor(decoded=["decoded-audio"]),
        )

        with (
            patch("tts_service.app.preload_engine_async", autospec=True),
            patch("tts_service.app.get_engine", return_value=fake_engine),
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

    def test_speech_endpoint_rejects_unknown_voice(self):
        with patch("tts_service.app.preload_engine_async", autospec=True):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": PUBLIC_BACKEND_TTS_MODEL,
                        "input": "hello world",
                        "voice": "not-a-real-voice",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("voice", response.json()["detail"])

    def test_speech_endpoint_rejects_upstream_repo_model_ids(self):
        with patch("tts_service.app.preload_engine_async", autospec=True):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "bosonai/higgs-audio-v2-generation-3B-base",
                        "input": "hello world",
                        "voice": "alloy",
                        "response_format": "pcm",
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn(PUBLIC_BACKEND_TTS_MODEL, response.json()["detail"])

    def test_healthz_returns_503_while_engine_is_loading(self):
        with (
            patch.dict("os.environ", {"HIGGS_HEALTH_REQUIRE_MODEL": "1"}),
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

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Higgs engine is still loading.")

    def test_healthz_returns_500_when_engine_preload_failed(self):
        with (
            patch.dict("os.environ", {"HIGGS_HEALTH_REQUIRE_MODEL": "1"}),
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

        self.assertEqual(response.status_code, 500)
        self.assertIn("boom", response.json()["detail"])

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
            patch.dict("os.environ", {}, clear=True),
            patch("tts_service.app.preload_engine_async", Mock()) as preload,
        ):
            tts_app.startup_preload()

        preload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
