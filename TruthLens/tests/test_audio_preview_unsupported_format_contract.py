import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioPreviewUnsupportedFormatContractTests(unittest.TestCase):
    def test_unsupported_extension_returns_415_without_running_ffmpeg(self):
        async def run_test():
            upload = UploadFile(
                filename="recording.xyz",
                file=io.BytesIO(b"not audio"),
            )

            with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(
                        api_server.BackgroundTasks(),
                        upload,
                    )

            self.assertEqual(context.exception.status_code, 415)
            self.assertIn("Unsupported audio file format", context.exception.detail)
            ffmpeg_lookup.assert_not_called()
            await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
