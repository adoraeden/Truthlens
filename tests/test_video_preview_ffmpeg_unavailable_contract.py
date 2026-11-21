import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

import src.api_server as api_server


class VideoPreviewFfmpegUnavailableContractTests(unittest.TestCase):
    def test_missing_ffmpeg_returns_503_and_closes_file(self):
        async def run_test():
            upload = UploadFile(filename="recording.mp4", file=io.BytesIO(b"video bytes"))
            with patch.object(api_server.shutil, "which", return_value=None):
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_video_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 503)
            self.assertIn("FFmpeg is not available", context.exception.detail)
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
