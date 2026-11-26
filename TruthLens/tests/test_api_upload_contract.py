import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

import src.api_server as api_server


class UploadStorageContractTests(unittest.TestCase):
    def test_save_upload_normalizes_uppercase_extension_and_preserves_bytes(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                upload = UploadFile(
                    filename="sample.WAV",
                    file=io.BytesIO(b"sample audio bytes"),
                )

                try:
                    with patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ):
                        saved_path = await api_server.save_upload(upload, "audio")

                    try:
                        self.assertEqual(saved_path.suffix, ".wav")
                        self.assertEqual(saved_path.read_bytes(), b"sample audio bytes")
                        self.assertEqual(saved_path.parent, Path(directory))
                    finally:
                        saved_path.unlink(missing_ok=True)
                finally:
                    await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
