import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AudioInvalidInputContractTests(unittest.TestCase):
    def test_audio_value_error_returns_422_and_cleans_upload(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                saved_path = Path(directory) / "truthlens_invalid_audio.wav"
                saved_path.write_bytes(b"audio bytes")
                upload = UploadFile(
                    filename="sample.wav",
                    file=io.BytesIO(b"audio bytes"),
                )

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
                        side_effect=ValueError("Audio waveform is invalid."),
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "audio")

                self.assertEqual(context.exception.status_code, 422)
                self.assertEqual(context.exception.detail, "Audio waveform is invalid.")
                self.assertFalse(saved_path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
