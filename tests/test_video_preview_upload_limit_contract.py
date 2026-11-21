import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

import src.api_server as api_server


class VideoPreviewUploadLimitContractTests(unittest.TestCase):
    def test_oversized_upload_returns_413_before_ffmpeg_lookup(self):
        async def run_test():
            upload = UploadFile(filename="large.mp4", file=io.BytesIO(b"12345"))
            with patch.object(api_server, "MAX_UPLOAD_BYTES", 4):
                with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_video_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 413)
            self.assertIn("100 MB upload limit", context.exception.detail)
            self.assertTrue(upload.file.closed)
            ffmpeg_lookup.assert_not_called()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
