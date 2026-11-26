import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import UploadFile

import src.api_server as api_server


class AudioAnalysisDispatchContractTests(unittest.TestCase):
    def test_audio_analysis_invokes_audio_detector_with_saved_path(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "audio.wav"
                saved_path.write_bytes(b"audio bytes")
                upload = UploadFile(filename="sample.wav", file=io.BytesIO(b"audio bytes"))
                result = {"verdict": "Likely real", "score": 0.12}
                detector = Mock()
                detector.predict.return_value = result

                async def execute(call):
                    return call()

                with (
                    patch.object(api_server, "save_upload", new_callable=AsyncMock, return_value=saved_path),
                    patch.dict(api_server.detectors, {"audio": detector}),
                    patch.object(api_server, "run_in_threadpool", new_callable=AsyncMock, side_effect=execute),
                ):
                    response = await api_server.analyze_upload(upload, "audio")

                detector.predict.assert_called_once_with(saved_path)
                self.assertEqual(response["result"], result)
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
