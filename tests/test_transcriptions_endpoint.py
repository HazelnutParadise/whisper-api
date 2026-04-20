import asyncio
import json
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
        stack.enter_context(
            patch.object(
                whisper_app,
                "ensure_runtime_assets_ready",
                return_value=None,
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

    def test_returns_202_when_model_download_starts(self):
        pending_response = whisper_app.ModelDownloadPendingResponse(
            status="model_downloading",
            detail="Requested model assets are downloading. Retry shortly.",
            resources=["asr:turbo"],
        )

        with (
            patch.object(
                whisper_app,
                "ensure_runtime_assets_ready",
                return_value=pending_response,
                create=True,
            ),
            patch.object(
                whisper_app,
                "save_upload_file",
                side_effect=AssertionError("upload should not be persisted yet"),
                create=True,
            ),
        ):
            response = asyncio.run(
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

        self.assertEqual(response.status_code, 202)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "model_downloading")
        self.assertEqual(payload["resources"], ["asr:turbo"])

    def test_simple_transcription_uses_legacy_whisper_model_alias(self):
        stack, route_paths = self.build_context({})
        with stack:
            with (
                patch.object(
                    FakeWhisperXModule,
                    "load_align_model",
                    side_effect=AssertionError(
                        "alignment should not run for simple responses"
                    ),
                ),
                patch.object(
                    FakeWhisperXModule,
                    "align",
                    side_effect=AssertionError(
                        "alignment should not run for simple responses"
                    ),
                ),
            ):
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

        payload_data = payload.model_dump()
        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertNotIn("/v1/audio/transcriptions/whisperx", route_paths)
        self.assertEqual(payload_data, {"text": "hello world"})

    def test_advanced_transcription_keeps_alignment_without_diarization(self):
        stack, route_paths = self.build_context({})
        with stack:
            module = FakeWhisperXModule()
            with (
                patch.object(
                    whisper_app,
                    "get_whisperx_module",
                    return_value=module,
                    create=True,
                ),
                patch.object(
                    whisper_app,
                    "get_align_model",
                    wraps=whisper_app.get_align_model,
                ) as get_align_model,
                patch.object(
                    module,
                    "align",
                    wraps=module.align,
                ) as align,
            ):
                payload = asyncio.run(
                    whisper_app.transcribe(
                        file=self.make_upload(),
                        model_name="whisper-1",
                        language=None,
                        advanced=True,
                        diarize=False,
                        min_speakers=None,
                        max_speakers=None,
                    )
                )

        payload_data = payload.model_dump()
        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertEqual(payload_data["text"], "hello world")
        self.assertEqual(payload_data["language"], "en")
        self.assertEqual(get_align_model.call_count, 1)
        self.assertEqual(align.call_count, 1)
        self.assertEqual(payload_data["diarization"], [])
        self.assertEqual(payload_data["speakers"], [])

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

        payload_data = payload.model_dump()
        self.assertIn("/v1/audio/transcriptions", route_paths)
        self.assertEqual(payload_data["text"], "hello world")
        self.assertEqual(payload_data["language"], "en")
        self.assertEqual(payload_data["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(
            payload_data["segments"][0]["words"][0]["speaker"],
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

    def test_openapi_describes_transcription_parameters_and_responses(self):
        schema = whisper_app.app.openapi()
        operation = schema["paths"]["/v1/audio/transcriptions"]["post"]
        request_body = operation["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        body_ref = request_body["$ref"].split("/")[-1]
        body_schema = schema["components"]["schemas"][body_ref]

        self.assertIn("advanced", body_schema["properties"])
        self.assertIn(
            "return only `{text}`",
            body_schema["properties"]["advanced"]["description"].lower(),
        )
        self.assertIn(
            "speaker diarization",
            body_schema["properties"]["diarize"]["description"].lower(),
        )
        self.assertIn(
            "whisper-1",
            body_schema["properties"]["model"]["description"].lower(),
        )

        response_schema = operation["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        self.assertIn("anyOf", response_schema)
        refs = [item["$ref"] for item in response_schema["anyOf"]]
        self.assertTrue(
            any(ref.endswith("TranscriptionSimpleResponse") for ref in refs)
        )
        self.assertTrue(
            any(ref.endswith("TranscriptionAdvancedResponse") for ref in refs)
        )


if __name__ == "__main__":
    unittest.main()
