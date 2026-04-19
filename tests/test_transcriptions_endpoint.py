import asyncio
import os
import tempfile
import unittest
from contextlib import ExitStack
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import app as whisper_app


class FakeWhisperXModel:
    def transcribe(
        self,
        audio,
        batch_size=None,
        language=None,
        task=None,
        chunk_size=30,
        **kwargs,
    ):
        del audio, batch_size, task, chunk_size, kwargs
        return {
            "language": language or "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.2,
                    "text": " hello world",
                }
            ],
        }


class FakeDiarizationPipeline:
    def __call__(
        self,
        audio,
        min_speakers=None,
        max_speakers=None,
        **kwargs,
    ):
        del audio, min_speakers, max_speakers, kwargs
        return [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.2},
        ]


class FakeWhisperXModule:
    def load_audio(self, path):
        return f"audio:{path}"

    def load_align_model(self, language_code, device):
        return (
            f"align-model:{language_code}:{device}",
            {"language_code": language_code},
        )

    def align(
        self,
        segments,
        model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    ):
        del model, metadata, audio, device, return_char_alignments
        return {
            "language": "en",
            "segments": [
                {
                    **segments[0],
                    "text": "hello world",
                    "words": [
                        {
                            "word": "hello",
                            "start": 0.0,
                            "end": 0.5,
                        },
                        {
                            "word": "world",
                            "start": 0.6,
                            "end": 1.2,
                        },
                    ],
                }
            ],
        }

    def assign_word_speakers(self, diarize_segments, result):
        result["segments"][0]["speaker"] = diarize_segments[0]["speaker"]
        for word in result["segments"][0]["words"]:
            word["speaker"] = diarize_segments[0]["speaker"]
        return result


class TranscriptionsEndpointTests(unittest.TestCase):
    def build_context(self, env):
        stack = ExitStack()
        tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, env, clear=False))
        stack.enter_context(
            patch.object(whisper_app, "UPLOAD_FOLDER", tmpdir, create=True)
        )
        stack.enter_context(
            patch.object(
                whisper_app,
                "is_ffmpeg_available",
                return_value=True,
                create=True,
            )
        )
        stack.enter_context(
            patch.object(whisper_app.shutil, "which", return_value="ffmpeg")
        )
        stack.enter_context(
            patch.object(
                whisper_app,
                "load_whisper_model",
                return_value=object(),
                create=True,
            )
        )
        stack.enter_context(
            patch.object(
                whisper_app,
                "load_whisperx_model",
                return_value=FakeWhisperXModel(),
                create=True,
            )
        )
        stack.enter_context(
            patch.object(
                whisper_app,
                "get_whisperx_module",
                return_value=FakeWhisperXModule(),
                create=True,
            )
        )
        if env.get("WHISPERX_HF_TOKEN") or env.get("HF_TOKEN"):
            stack.enter_context(
                patch.object(
                    whisper_app,
                    "get_diarization_pipeline",
                    return_value=FakeDiarizationPipeline(),
                    create=True,
                )
            )
        route_paths = {
            route.path for route in whisper_app.app.routes if hasattr(route, "path")
        }
        return stack, route_paths

    def make_upload(self):
        return UploadFile(filename="sample.wav", file=BytesIO(b"audio"))

    def test_simple_transcription_uses_legacy_whisper_model_alias(self):
        stack, route_paths = self.build_context({})
        with stack:
            payload = asyncio.run(
                whisper_app.transcribe(
                    file=self.make_upload(),
                    model_name="whisper-1",
                    language=None,
                    advanced=False,
                    diarize=True,
                    min_speakers=None,
                    max_speakers=None,
                )
            )

        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertNotIn("/v1/audio/transcriptions/whisperx", route_paths)
        self.assertEqual(payload, {"text": "hello world"})

    def test_advanced_transcription_returns_timestamps_and_speakers(self):
        stack, route_paths = self.build_context(
            {"WHISPERX_HF_TOKEN": "hf-test-token"}
        )
        with stack:
            payload = asyncio.run(
                whisper_app.transcribe(
                    file=self.make_upload(),
                    model_name="whisper-1",
                    language=None,
                    advanced=True,
                    diarize=True,
                    min_speakers=None,
                    max_speakers=None,
                )
            )

        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(
            payload["segments"][0]["words"][0]["speaker"],
            "SPEAKER_00",
        )

    def test_advanced_transcription_requires_token_for_diarization(self):
        stack, route_paths = self.build_context({})
        with stack:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    whisper_app.transcribe(
                        file=self.make_upload(),
                        model_name="whisper-1",
                        language=None,
                        advanced=True,
                        diarize=True,
                        min_speakers=None,
                        max_speakers=None,
                    )
                )

        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("token", str(raised.exception.detail).lower())

    def test_rejects_unsupported_model(self):
        stack, route_paths = self.build_context({})
        with stack:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    whisper_app.transcribe(
                        file=self.make_upload(),
                        model_name="large-v3",
                        language=None,
                        advanced=False,
                        diarize=False,
                        min_speakers=None,
                        max_speakers=None,
                    )
                )

        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("whisper-1", str(raised.exception.detail))
        self.assertIn("turbo", str(raised.exception.detail))


if __name__ == "__main__":
    unittest.main()
