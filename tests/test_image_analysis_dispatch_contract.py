import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import UploadFile

import src.api_server as api_server


class ImageAnalysisDispatchContractTests(unittest.TestCase):
    def test_image_analysis_invokes_image_detector_with_saved_path(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "image.png"
                saved_path.write_bytes(b"image bytes")
                upload = UploadFile(filename="sample.png", file=io.BytesIO(b"image bytes"))
                result = {"verdict": "Likely real", "image_size": (320, 240)}
                detector = Mock()
                detector.predict.return_value = result

                async def execute(call):
                    return call()

                with (
                    patch.object(api_server, "save_upload", new_callable=AsyncMock, return_value=saved_path),
                    patch.dict(api_server.detectors, {"image": detector}),
                    patch.object(api_server, "run_in_threadpool", new_callable=AsyncMock, side_effect=execute),
                ):
                    response = await api_server.analyze_upload(upload, "image")

                detector.predict.assert_called_once_with(saved_path)
                self.assertEqual(response["result"]["image_size"], [320, 240])
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
