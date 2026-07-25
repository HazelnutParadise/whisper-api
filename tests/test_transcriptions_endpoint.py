import asyncio
import os
import tempfile
import threading
import unittest
from contextlib import ExitStack
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import whisper_service.app as whisper_app


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
        if env.get("HF_TOKEN"):
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

    def test_ensure_runtime_asset_singleflight_avoids_duplicate_loads(self):
        with whisper_app.runtime_asset_lock:
            whisper_app.runtime_asset_states.clear()
            whisper_app.runtime_asset_errors.clear()
            whisper_app.runtime_asset_events.clear()

        started = threading.Event()
        release = threading.Event()
        load_count = {"value": 0}

        def loader():
            load_count["value"] += 1
            started.set()
            release.wait(timeout=2)

        first = threading.Thread(
            target=lambda: whisper_app.ensure_runtime_asset(
                resource="asr:turbo",
                is_ready=lambda: load_count["value"] > 0,
                loader=loader,
            )
        )
        first.start()
        self.assertTrue(started.wait(timeout=2))

        whisper_app.ensure_runtime_asset(
            resource="asr:turbo",
            is_ready=lambda: load_count["value"] > 0,
            loader=loader,
        )
        release.set()
        first.join(timeout=2)

        self.assertEqual(load_count["value"], 1)

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
                patch.object(
                    whisper_app,
                    "unload_whisperx_runtime",
                    return_value=None,
                ) as unload_runtime,
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
        unload_runtime.assert_called_once()

    def test_transcription_can_keep_runtime_loaded_when_configured(self):
        stack, _ = self.build_context({"WHISPERX_UNLOAD_AFTER_REQUEST": "0"})
        with stack:
            with patch.object(
                whisper_app,
                "unload_whisperx_runtime",
                return_value=None,
            ) as unload_runtime:
                payload = asyncio.run(
                    whisper_app.transcribe(
                        file=self.make_upload(),
                        model_name="whisper-1",
                        language=None,
                        advanced=False,
                        diarize=False,
                        min_speakers=None,
                        max_speakers=None,
                    )
                )

        self.assertEqual(payload.model_dump(), {"text": "hello world"})
        unload_runtime.assert_not_called()

    def test_unload_whisperx_runtime_clears_model_and_asset_caches(self):
        whisper_app.whisperx_models["whisper-1"] = object()
        whisper_app.whisperx_align_models["en"] = (object(), {})
        whisper_app.whisperx_diarization_pipeline = object()
        with whisper_app.runtime_asset_lock:
            whisper_app.runtime_asset_states["asr:turbo"] = "ready"
            whisper_app.runtime_asset_errors["asr:turbo"] = "boom"
            whisper_app.runtime_asset_events["asr:turbo"] = threading.Event()

        whisper_app.unload_whisperx_runtime()

        self.assertEqual(whisper_app.whisperx_models, {})
        self.assertEqual(whisper_app.whisperx_align_models, {})
        self.assertIsNone(whisper_app.whisperx_diarization_pipeline)
        self.assertEqual(whisper_app.runtime_asset_states, {})
        self.assertEqual(whisper_app.runtime_asset_errors, {})
        self.assertEqual(whisper_app.runtime_asset_events, {})

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
        stack, route_paths = self.build_context({"HF_TOKEN": "hf-test-token"})
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

    def test_chunk_speaker_mapping_aligns_local_ids_to_previous_chunk(self):
        previous_segments = [
            whisper_app.SegmentTimestamp(
                id=0,
                start=57.0,
                end=60.0,
                text="previous",
                speaker="SPEAKER_00",
                words=None,
            )
        ]
        current_segments = [
            whisper_app.SegmentTimestamp(
                id=0,
                start=58.0,
                end=61.0,
                text="current",
                speaker="SPEAKER_01",
                words=[
                    whisper_app.WordTimestamp(
                        word="current",
                        start=58.2,
                        end=58.8,
                        speaker="SPEAKER_01",
                    )
                ],
            )
        ]

        mapping = whisper_app.build_chunk_speaker_mapping(
            previous_segments=previous_segments,
            current_segments=current_segments,
            boundary_start=60.0,
            overlap_seconds=3.0,
            known_speakers=["SPEAKER_00"],
            min_similarity=0.1,
        )
        whisper_app.apply_speaker_mapping_to_segments(current_segments, mapping)

        self.assertEqual(mapping["SPEAKER_01"], "SPEAKER_00")
        self.assertEqual(current_segments[0].speaker, "SPEAKER_00")
        self.assertEqual(current_segments[0].words[0].speaker, "SPEAKER_00")

    def test_chunk_speaker_mapping_respects_min_similarity_threshold(self):
        previous_segments = [
            whisper_app.SegmentTimestamp(
                id=0,
                start=57.0,
                end=60.0,
                text="previous",
                speaker="SPEAKER_00",
                words=None,
            )
        ]
        current_segments = [
            whisper_app.SegmentTimestamp(
                id=0,
                start=59.8,
                end=61.0,
                text="current",
                speaker="SPEAKER_99",
                words=None,
            )
        ]

        mapping = whisper_app.build_chunk_speaker_mapping(
            previous_segments=previous_segments,
            current_segments=current_segments,
            boundary_start=60.0,
            overlap_seconds=3.0,
            known_speakers=["SPEAKER_00"],
            min_similarity=0.95,
        )

        self.assertEqual(mapping["SPEAKER_99"], "SPEAKER_01")

    def test_chunking_decision_reads_audio_duration_not_file_size(self):
        # A 16 kHz mono mp3 holds roughly 29 MB per hour, so a byte threshold
        # waves multi-hour recordings through unchunked while splitting a short
        # lossless upload for no reason. Duration is what the pipeline actually
        # struggles with, so that is what the decision reads.
        with patch.object(
            whisper_app, "get_audio_duration_seconds", return_value=7200.0
        ):
            self.assertTrue(
                whisper_app.should_use_chunked_transcription("small-but-long.mp3")
            )

        with patch.object(
            whisper_app, "get_audio_duration_seconds", return_value=60.0
        ):
            self.assertFalse(
                whisper_app.should_use_chunked_transcription("large-but-short.wav")
            )

    def test_chunking_decision_falls_back_to_single_pass_when_ffprobe_fails(self):
        with patch.object(
            whisper_app,
            "get_audio_duration_seconds",
            side_effect=HTTPException(status_code=500, detail="unreadable"),
        ):
            self.assertFalse(
                whisper_app.should_use_chunked_transcription("unreadable.bin")
            )

    def test_advanced_transcription_uses_chunked_pipeline_only_when_enabled(self):
        stack, _ = self.build_context({"HF_TOKEN": "hf-test-token"})
        chunked_payload = whisper_app.TranscriptionAdvancedResponse(
            text="chunked",
            language="en",
            segments=[],
            diarization=[],
            speakers=[],
        )

        with stack:
            with (
                patch.object(
                    whisper_app,
                    "should_use_chunked_transcription",
                    return_value=True,
                ),
                patch.object(
                    whisper_app,
                    "build_chunked_whisperx_response",
                    return_value=chunked_payload,
                ) as build_chunked,
                patch.object(
                    whisper_app,
                    "build_whisperx_response",
                    side_effect=AssertionError("non-chunked path should not run"),
                ),
            ):
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

        self.assertEqual(payload.model_dump()["text"], "chunked")
        self.assertEqual(build_chunked.call_count, 1)

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
        self.assertIn("HF_TOKEN", str(raised.exception.detail))

    def test_legacy_whisperx_token_is_no_longer_accepted(self):
        stack, _ = self.build_context(
            {"WHISPERX_HF_TOKEN": "hf-legacy-token", "HF_TOKEN": ""}
        )
        with stack:
            token = whisper_app.get_diarization_token()

        self.assertIsNone(token)

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
