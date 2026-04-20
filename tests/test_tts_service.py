import unittest
import wave
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from tts_service.app import (
    DEFAULT_BACKEND_TTS_MODEL,
    DEFAULT_SAMPLING_RATE,
    EngineBundle,
    OPENAI_VOICE_ALIASES,
    app,
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

        with patch("tts_service.app.get_engine", return_value=fake_engine):
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
        self.assertEqual(response.content, np.array([0, 16384, -16384], dtype=np.int16).tobytes())
        self.assertEqual(fake_processor.apply_calls[0][0][0][0]["role"], "system")
        self.assertEqual(fake_model.calls[0]["input_ids"], "fake-input-ids")
        self.assertEqual(fake_processor.decode_calls, [fake_outputs])

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
