import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewFfmpegUnavailableContractTests(unittest.TestCase):
    def test_missing_ffmpeg_returns_503_and_closes_file(self):
        async def run_test():
            upload = UploadFile(
                filename="recording.wav",
                file=io.BytesIO(b"audio bytes"),
            )

            with patch.object(api_server.shutil, "which", return_value=None):
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(
                        api_server.BackgroundTasks(),
                        upload,
                    )

            self.assertEqual(context.exception.status_code, 503)
            self.assertIn("FFmpeg is not available", context.exception.detail)
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
