import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class EmptyUploadContractTests(unittest.TestCase):
    def test_empty_upload_is_rejected_before_temp_file_creation(self):
        async def run_test():
            upload = UploadFile(filename="empty.wav", file=io.BytesIO(b""))
            with patch.object(api_server.tempfile, "NamedTemporaryFile") as create_temp:
                with self.assertRaises(HTTPException) as context:
                    await api_server.save_upload(upload, "audio")

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("empty", context.exception.detail.lower())
            create_temp.assert_not_called()
            await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
