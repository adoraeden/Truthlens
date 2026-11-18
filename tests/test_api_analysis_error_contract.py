import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AnalysisErrorContractTests(unittest.TestCase):
    def test_invalid_detector_input_returns_422_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_invalid_image.png"
                saved_path.write_bytes(b"test image bytes")
                upload = UploadFile(
                    filename="invalid.png",
                    file=io.BytesIO(b"test image bytes"),
                )

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
                        side_effect=ValueError("Image data could not be decoded."),
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "image")

                self.assertEqual(context.exception.status_code, 422)
                self.assertEqual(
                    context.exception.detail,
                    "Image data could not be decoded.",
                )
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())

    def test_unavailable_audio_detector_returns_503_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_unavailable_audio.wav"
                saved_path.write_bytes(b"test audio bytes")
                upload = UploadFile(
                    filename="sample.wav",
                    file=io.BytesIO(b"test audio bytes"),
                )
                unavailable_reason = "PellaV2 is unavailable because of the RAM limit."

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=saved_path,
                    ),
                    patch.dict(api_server.detectors, {"audio": None}),
                    patch.object(
                        api_server,
                        "audio_unavailable_reason",
                        unavailable_reason,
                    ),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                    ) as run_inference,
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "audio")

                self.assertEqual(context.exception.status_code, 503)
                self.assertEqual(context.exception.detail, unavailable_reason)
                run_inference.assert_not_awaited()
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
