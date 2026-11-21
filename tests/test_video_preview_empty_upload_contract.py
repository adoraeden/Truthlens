import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

import src.api_server as api_server


class VideoPreviewEmptyUploadContractTests(unittest.TestCase):
    def test_empty_upload_returns_400_and_closes_file(self):
        async def run_test():
            upload = UploadFile(filename="empty.mp4", file=io.BytesIO(b""))
            with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_video_preview(BackgroundTasks(), upload)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("empty", context.exception.detail.lower())
            self.assertTrue(upload.file.closed)
            ffmpeg_lookup.assert_not_called()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
