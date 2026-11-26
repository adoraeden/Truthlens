import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class UploadSizeLimitContractTests(unittest.TestCase):
    def test_oversized_upload_reads_only_one_byte_beyond_limit(self):
        async def run_test():
            upload = UploadFile(filename="large.wav", file=io.BytesIO(b"12345"))
            with (
                patch.object(api_server, "MAX_UPLOAD_BYTES", 4),
                patch.object(upload, "read", new_callable=AsyncMock, return_value=b"12345") as read,
                patch.object(api_server.tempfile, "NamedTemporaryFile") as create_temp,
            ):
                with self.assertRaises(HTTPException) as context:
                    await api_server.save_upload(upload, "audio")

            self.assertEqual(context.exception.status_code, 413)
            read.assert_awaited_once_with(5)
            create_temp.assert_not_called()
            await upload.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
