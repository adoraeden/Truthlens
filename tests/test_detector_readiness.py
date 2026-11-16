import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class DetectorReadinessTests(unittest.TestCase):
    def test_health_reports_detector_readiness_and_audio_unavailable_reason(self):
        reason = "PellaV2 is unavailable because the server has insufficient free memory."

        with (
            patch.dict(
                api_server.detectors,
                {"image": None, "audio": None, "video": None},
            ),
            patch.object(api_server, "audio_unavailable_reason", reason),
        ):
            response = api_server.health()

        self.assertEqual(response["status"], "offline")
        self.assertEqual(
            response["detectors"],
            {"image": False, "audio": False, "video": False},
        )
        self.assertEqual(response["audio_unavailable_reason"], reason)

    def test_audio_request_returns_503_and_cleans_upload_when_detector_is_unavailable(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_unavailable.wav"
                saved_path.write_bytes(b"test audio")
                reason = "PellaV2 is unavailable because of the RAM limit."
                upload = UploadFile(
                    filename="sample.wav",
                    file=io.BytesIO(b"test audio"),
                )

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=saved_path,
                    ),
                    patch.dict(api_server.detectors, {"audio": None}),
                    patch.object(api_server, "audio_unavailable_reason", reason),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "audio")

                self.assertEqual(context.exception.status_code, 503)
                self.assertEqual(context.exception.detail, reason)
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
