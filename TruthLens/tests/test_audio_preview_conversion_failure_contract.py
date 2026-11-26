import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewConversionFailureContractTests(unittest.TestCase):
    def test_ffmpeg_failure_returns_422_and_removes_temporary_files(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                failed_process = type(
                    "FailedProcess",
                    (),
                    {"returncode": 1, "stderr": b"simulated conversion failure"},
                )()
                upload = UploadFile(
                    filename="broken.wav",
                    file=io.BytesIO(b"audio bytes"),
                )

                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ),
                    patch.object(
                        api_server.subprocess,
                        "run",
                        return_value=failed_process,
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(
                            api_server.BackgroundTasks(),
                            upload,
                        )

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("Audio preview conversion failed", context.exception.detail)
                self.assertTrue(upload.file.closed)
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
