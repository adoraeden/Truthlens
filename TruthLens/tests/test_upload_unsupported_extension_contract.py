import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class UploadExtensionContractTests(unittest.TestCase):
    def test_unsupported_extension_is_rejected_before_reading_upload(self):
        async def run_test():
            upload = UploadFile(filename="sample.exe", file=io.BytesIO(b"payload"))
            with patch.object(upload, "read", new_callable=AsyncMock) as read:
                with self.assertRaises(HTTPException) as context:
                    await api_server.save_upload(upload, "image")

            self.assertEqual(context.exception.status_code, 415)
            read.assert_not_awaited()
            await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
