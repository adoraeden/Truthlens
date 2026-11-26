import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile

import src.api_server as api_server


class VideoAnalysisSuccessContractTests(unittest.TestCase):
    def test_successful_video_analysis_returns_result_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_video_success.mp4"
                saved_path.write_bytes(b"video bytes")
                upload = UploadFile(
                    filename="sample.mp4",
                    file=io.BytesIO(b"video bytes"),
                )
                detector_result = {
                    "verdict": "Likely manipulated",
                    "frames_analyzed": 8,
                }

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=saved_path,
                    ),
                    patch.dict(api_server.detectors, {"video": object()}),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                        return_value=detector_result,
                    ),
                ):
                    response = await api_server.analyze_upload(upload, "video")

                self.assertEqual(response["success"], True)
                self.assertEqual(response["modality"], "video")
                self.assertEqual(response["filename"], "sample.mp4")
                self.assertEqual(response["result"], detector_result)
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
