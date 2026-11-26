import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile

import src.api_server as api_server


class AudioAnalysisSuccessContractTests(unittest.TestCase):
    def test_successful_audio_analysis_returns_result_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_audio_success.wav"
                saved_path.write_bytes(b"audio bytes")
                upload = UploadFile(
                    filename="sample.wav",
                    file=io.BytesIO(b"audio bytes"),
                )
                detector_result = {"verdict": "Likely real", "confidence": 0.91}

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=saved_path,
                    ),
                    patch.dict(api_server.detectors, {"audio": object()}),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                        return_value=detector_result,
                    ),
                ):
                    response = await api_server.analyze_upload(upload, "audio")

                self.assertEqual(response["success"], True)
                self.assertEqual(response["modality"], "audio")
                self.assertEqual(response["filename"], "sample.wav")
                self.assertEqual(response["result"], detector_result)
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
