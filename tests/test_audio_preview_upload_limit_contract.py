import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewUploadLimitContractTests(unittest.TestCase):
    def test_oversized_upload_returns_413_and_skips_ffmpeg(self):
        async def run_test():
            upload = UploadFile(
                filename="large.wav",
                file=io.BytesIO(b"12345"),
            )

            with (
                patch.object(api_server, "MAX_UPLOAD_BYTES", 4),
                patch.object(api_server.shutil, "which") as ffmpeg_lookup,
            ):
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(
                        api_server.BackgroundTasks(),
                        upload,
                    )

            self.assertEqual(context.exception.status_code, 413)
            self.assertEqual(
                context.exception.detail,
                "File exceeds the 100 MB upload limit.",
            )
            ffmpeg_lookup.assert_not_called()
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
