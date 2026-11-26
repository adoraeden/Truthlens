import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewEmptyUploadContractTests(unittest.TestCase):
    def test_empty_upload_returns_400_and_closes_file(self):
        async def run_test():
            upload = UploadFile(
                filename="empty.wav",
                file=io.BytesIO(b""),
            )

            with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(
                        api_server.BackgroundTasks(),
                        upload,
                    )

            self.assertEqual(context.exception.status_code, 400)
            self.assertEqual(context.exception.detail, "The uploaded file is empty.")
            self.assertTrue(upload.file.closed)
            ffmpeg_lookup.assert_not_called()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
