import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

import src.api_server as api_server


class VideoPreviewTimeoutCleanupContractTests(unittest.TestCase):
    def test_ffmpeg_timeout_returns_422_and_removes_temporary_files(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                upload = UploadFile(filename="slow.mp4", file=io.BytesIO(b"video bytes"))
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
                        side_effect=api_server.subprocess.TimeoutExpired("ffmpeg", 180),
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_video_preview(BackgroundTasks(), upload)

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("timed out", context.exception.detail.lower())
                self.assertEqual(list(Path(directory).iterdir()), [])
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
