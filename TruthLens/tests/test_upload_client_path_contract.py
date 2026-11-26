import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

import src.api_server as api_server


class UploadClientPathContractTests(unittest.TestCase):
    def test_client_path_components_do_not_become_temp_file_name(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original = tempfile.NamedTemporaryFile

                def create_temp_file(**kwargs):
                    return original(dir=directory, **kwargs)

                upload = UploadFile(
                    filename="../../private/sample.WAV",
                    file=io.BytesIO(b"audio payload"),
                )
                try:
                    with patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ):
                        saved_path = await api_server.save_upload(upload, "audio")

                    try:
                        self.assertEqual(saved_path.parent, Path(directory))
                        self.assertEqual(saved_path.suffix, ".wav")
                        self.assertEqual(saved_path.read_bytes(), b"audio payload")
                    finally:
                        saved_path.unlink(missing_ok=True)
                finally:
                    await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
