import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from tts_service.app import (
    DEFAULT_BACKEND_TTS_MODEL,
    OPENAI_VOICE_ALIASES,
    app,
)


class FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class TTSServiceTests(unittest.TestCase):
    def test_openai_voice_aliases_map_to_supported_voice_prompts(self):
        self.assertEqual(OPENAI_VOICE_ALIASES["alloy"], "belinda")
        self.assertEqual(OPENAI_VOICE_ALIASES["ash"], "en_man")
        self.assertEqual(OPENAI_VOICE_ALIASES["shimmer"], "en_woman")

    def test_speech_endpoint_returns_pcm_audio_from_native_engine(self):
        fake_response = SimpleNamespace(
            audio=np.array([0.0, 0.5, -0.5], dtype=np.float32),
            sampling_rate=24_000,
        )
        fake_engine = FakeEngine(fake_response)

        with (
            patch("tts_service.app.get_engine", return_value=fake_engine),
            patch("tts_service.app.build_chat_ml_sample", return_value="sample"),
        ):
            client = TestClient(app)
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
        self.assertEqual(response.headers["content-type"], "audio/pcm")
        self.assertEqual(len(response.content), 6)
        self.assertEqual(fake_engine.calls[0]["chat_ml_sample"], "sample")
        self.assertTrue(fake_engine.calls[0]["force_audio_gen"])

    def test_speech_endpoint_rejects_unknown_voice(self):
        client = TestClient(app)

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_BACKEND_TTS_MODEL,
                "input": "hello world",
                "voice": "not-a-real-voice",
                "response_format": "pcm",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("voice", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
