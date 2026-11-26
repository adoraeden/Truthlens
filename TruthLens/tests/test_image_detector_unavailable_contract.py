import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class ImageDetectorUnavailableContractTests(unittest.TestCase):
    def test_unavailable_image_detector_returns_503_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "unavailable.png"
                saved_path.write_bytes(b"image bytes")
                upload = UploadFile(filename="sample.png", file=io.BytesIO(b"image bytes"))
                with (
                    patch.object(api_server, "save_upload", new_callable=AsyncMock, return_value=saved_path),
                    patch.dict(api_server.detectors, {"image": None}),
                    patch.object(api_server, "run_in_threadpool", new_callable=AsyncMock) as inference,
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "image")

                self.assertEqual(context.exception.status_code, 503)
                self.assertEqual(context.exception.detail, "The image detector is not ready.")
                inference.assert_not_awaited()
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
