import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile

import src.api_server as api_server


class ImageAnalysisResponseTests(unittest.TestCase):
    def test_image_dimensions_are_json_compatible_and_upload_is_cleaned(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_image_test.png"
                saved_path.write_bytes(b"test image bytes")
                upload = UploadFile(
                    filename="sample.png",
                    file=io.BytesIO(b"test image bytes"),
                )
                detector_result = {
                    "verdict": "Likely real",
                    "image_size": (640, 480),
                }

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=saved_path,
                    ),
                    patch.dict(api_server.detectors, {"image": object()}),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                        return_value=detector_result,
                    ),
                ):
                    response = await api_server.analyze_upload(upload, "image")

                self.assertEqual(response["modality"], "image")
                self.assertEqual(response["filename"], "sample.png")
                self.assertEqual(response["result"]["image_size"], [640, 480])
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
