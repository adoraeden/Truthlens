import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewConversionTests(unittest.TestCase):
    """Regression tests for the browser-only audio preview conversion endpoint."""

    def make_upload(self, filename="recording.wav", data=b"audio bytes"):
        return UploadFile(filename=filename, file=io.BytesIO(data))

    def make_temp_file_factory(self, directory):
        original_named_tempfile = tempfile.NamedTemporaryFile

        def create_temp_file(**kwargs):
            return original_named_tempfile(dir=directory, **kwargs)

        return create_temp_file

    def test_rejects_unsupported_extension(self):
        async def run_test():
            upload = self.make_upload("recording.xyz", b"not audio")
            with self.assertRaises(HTTPException) as context:
                await api_server.convert_audio_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 415)
            self.assertIn("Unsupported audio file format", context.exception.detail)

        asyncio.run(run_test())

    def test_rejects_empty_upload_and_closes_file(self):
        async def run_test():
            upload = self.make_upload("empty.wav", b"")
            with self.assertRaises(HTTPException) as context:
                await api_server.convert_audio_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("empty", context.exception.detail.lower())
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())

    def test_returns_503_when_ffmpeg_is_unavailable(self):
        async def run_test():
            upload = self.make_upload()
            with patch.object(api_server.shutil, "which", return_value=None):
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 503)
            self.assertIn("FFmpeg is not available", context.exception.detail)
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())

    def test_rejects_upload_over_size_limit_before_checking_ffmpeg(self):
        async def run_test():
            upload = self.make_upload("large.wav", b"12345")
            with patch.object(api_server, "MAX_UPLOAD_BYTES", 4):
                with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 413)
            self.assertIn("100 MB upload limit", context.exception.detail)
            ffmpeg_lookup.assert_not_called()
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())

    def test_ffmpeg_failure_cleans_temporary_source_and_output(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                upload = self.make_upload("broken.wav")
                failed_ffmpeg = type(
                    "FailedProcess",
                    (),
                    {"returncode": 1, "stderr": b"simulated ffmpeg failure"},
                )()

                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=self.make_temp_file_factory(directory),
                    ),
                    patch.object(api_server.subprocess, "run", return_value=failed_ffmpeg),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(BackgroundTasks(), upload)

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("Audio preview conversion failed", context.exception.detail)
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())

    def test_ffmpeg_timeout_cleans_temporary_source_and_output(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                upload = self.make_upload("slow.wav")
                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=self.make_temp_file_factory(directory),
                    ),
                    patch.object(
                        api_server.subprocess,
                        "run",
                        side_effect=api_server.subprocess.TimeoutExpired("ffmpeg", 120),
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(BackgroundTasks(), upload)

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("timed out", context.exception.detail.lower())
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())

    def test_success_returns_mp3_and_defers_output_cleanup_until_response_finishes(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                upload = self.make_upload("recording.flac", b"original audio")
                background_tasks = BackgroundTasks()
                captured = {}

                def fake_ffmpeg(command, **kwargs):
                    captured["command"] = command
                    Path(command[-1]).write_bytes(b"generated mp3 preview")
                    return type(
                        "CompletedProcess",
                        (),
                        {"returncode": 0, "stderr": b""},
                    )()

                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=self.make_temp_file_factory(directory),
                    ),
                    patch.object(api_server.subprocess, "run", side_effect=fake_ffmpeg),
                ):
                    response = await api_server.convert_audio_preview(
                        background_tasks,
                        upload,
                    )

                command = captured["command"]
                self.assertIn("-vn", command)
                self.assertEqual(command[command.index("-ac") + 1], "2")
                self.assertEqual(command[command.index("-ar") + 1], "44100")
                self.assertEqual(command[command.index("-b:a") + 1], "128k")
                self.assertEqual(response.media_type, "audio/mpeg")
                self.assertTrue(Path(response.path).is_file())
                self.assertEqual(
                    [path.name for path in Path(directory).iterdir()],
                    [Path(response.path).name],
                )

                await background_tasks()
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())

    def test_missing_ffmpeg_output_returns_422_and_cleans_source(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                upload = self.make_upload("recording.wav")
                successful_but_empty_ffmpeg = type(
                    "CompletedProcess",
                    (),
                    {"returncode": 0, "stderr": b""},
                )()

                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=self.make_temp_file_factory(directory),
                    ),
                    patch.object(
                        api_server.subprocess,
                        "run",
                        return_value=successful_but_empty_ffmpeg,
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(BackgroundTasks(), upload)

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("produced no output", context.exception.detail)
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
