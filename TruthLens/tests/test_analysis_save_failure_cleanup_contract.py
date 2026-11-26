import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

import src.api_server as api_server


class AnalysisSaveFailureCleanupTests(unittest.TestCase):
    def test_upload_is_closed_when_storage_validation_fails(self):
        async def run_test():
            upload = UploadFile(filename="bad.exe", file=io.BytesIO(b"payload"))
            with patch.object(
                api_server,
                "save_upload",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=415, detail="unsupported"),
            ):
                with self.assertRaises(HTTPException) as context:
                    await api_server.analyze_upload(upload, "image")

            self.assertEqual(context.exception.status_code, 415)
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
