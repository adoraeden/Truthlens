import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

import src.api_server as api_server


class UploadWriteFailureCleanupTests(unittest.TestCase):
    def test_partial_temp_file_is_removed_when_write_fails(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original = tempfile.NamedTemporaryFile

                class FailingTempFile:
                    def __init__(self, **kwargs):
                        self.handle = original(dir=directory, **kwargs)
                        self.name = self.handle.name

                    def __enter__(self):
                        self.handle.__enter__()
                        return self

                    def __exit__(self, *args):
                        return self.handle.__exit__(*args)

                    def write(self, data):
                        self.handle.write(data[:2])
                        raise OSError("simulated disk write failure")

                upload = UploadFile(filename="sample.wav", file=io.BytesIO(b"payload"))
                with patch.object(
                    api_server.tempfile,
                    "NamedTemporaryFile",
                    side_effect=lambda **kwargs: FailingTempFile(**kwargs),
                ):
                    with self.assertRaises(OSError):
                        await api_server.save_upload(upload, "audio")

                self.assertEqual(list(Path(directory).iterdir()), [])
                await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
