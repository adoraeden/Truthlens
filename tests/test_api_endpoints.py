import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import src.api_server as api_server
from src.api_server import app, detectors


class ApiEndpointTests(unittest.TestCase):
    def test_health_endpoint_contract(self):
        with TestClient(app) as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("status", data)
            self.assertIn("detectors", data)
            self.assertIn("image", data["detectors"])
            self.assertIn("audio", data["detectors"])
            self.assertIn("video", data["detectors"])

    def test_unsupported_audio_extension_returns_415(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/detect/audio",
                files={"file": ("test.unsupported", io.BytesIO(b"data"), "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 415)
            self.assertIn("Unsupported audio file format", response.json()["detail"])

    def test_unsupported_video_extension_returns_415(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/detect/video",
                files={"file": ("test.unsupported", io.BytesIO(b"data"), "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 415)
            self.assertIn("Unsupported video file format", response.json()["detail"])

    def test_empty_audio_upload_returns_400(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/detect/audio",
                files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("uploaded file is empty", response.json()["detail"])

    def test_audio_endpoint_with_real_wav(self):
        real_wav = Path("test_audio_wav/real1.wav")
        if not real_wav.is_file():
            self.skipTest("real1.wav not found")

        with TestClient(app) as client:
            if detectors["audio"] is None:
                self.skipTest("audio detector not loaded (RAM constraint)")

            with open(real_wav, "rb") as f:
                response = client.post(
                    "/api/detect/audio",
                    files={"file": ("real1.wav", f, "audio/wav")},
                )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["modality"], "audio")
            self.assertEqual(body["result"]["verdict"], "Likely real")
            self.assertLess(body["result"]["score"], 0.50)
            self.assertEqual(body["result"]["duration_analyzed_seconds"], 4.0)
            self.assertEqual(body["result"]["analysis_region"], "center_crop")

    def test_video_endpoint_with_sample_video(self):
        test_video = Path("test_videos/Golgappe-Ka-SnapYT.App.mp4")
        if not test_video.is_file():
            self.skipTest("Sample video not found")

        with TestClient(app) as client:
            if detectors["video"] is None or detectors["image"] is None:
                self.skipTest("image/video detector not loaded")

            with open(test_video, "rb") as f:
                response = client.post(
                    "/api/detect/video",
                    files={"file": ("test.mp4", f, "video/mp4")},
                )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["modality"], "video")
            res = body["result"]
            self.assertGreater(res["frames_analyzed"], 0)
            self.assertIn("frame_results", res)
            self.assertIn("audio_status", res)
            self.assertIn(res["audio_status"], ["analyzed", "no_audio_stream", "audio_detector_unavailable"])

    def test_video_preview_conversion_endpoint(self):
        test_video = Path("test_videos/Golgappe-Ka-SnapYT.App.mp4")
        if not test_video.is_file():
            self.skipTest("Sample video not found")

        with TestClient(app) as client:
            with open(test_video, "rb") as f:
                response = client.post(
                    "/api/convert/video-preview",
                    files={"file": ("test.mp4", f, "video/mp4")},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "video/mp4")
            self.assertGreater(len(response.content), 1000)

    def test_video_preview_unsupported_format_returns_415(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/convert/video-preview",
                files={
                    "file": (
                        "test.xyz",
                        io.BytesIO(b"data"),
                        "application/octet-stream",
                    )
                },
            )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported video file format", response.json()["detail"])

    def test_video_preview_conversion_failure_returns_422_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            original_named_tempfile = tempfile.NamedTemporaryFile

            def create_temp_file(**kwargs):
                return original_named_tempfile(dir=directory, **kwargs)

            failed_ffmpeg = type(
                "FailedProcess",
                (),
                {"returncode": 1, "stderr": b"simulated ffmpeg failure"},
            )()

            with (
                patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                patch.object(
                    api_server.tempfile,
                    "NamedTemporaryFile",
                    side_effect=create_temp_file,
                ),
                patch.object(
                    api_server.subprocess,
                    "run",
                    return_value=failed_ffmpeg,
                ),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/api/convert/video-preview",
                    files={
                        "file": (
                            "broken.mp4",
                            io.BytesIO(b"not a real video"),
                            "video/mp4",
                        )
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertIn(
                "Video preview conversion failed",
                response.json()["detail"],
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_upload_over_size_limit_returns_413(self):
        async def run_test():
            upload = UploadFile(
                filename="large.wav",
                file=io.BytesIO(b"12345"),
            )

            with patch.object(api_server, "MAX_UPLOAD_BYTES", 4):
                with self.assertRaises(HTTPException) as context:
                    await api_server.save_upload(upload, "audio")

            self.assertEqual(context.exception.status_code, 413)
            await upload.close()

        asyncio.run(run_test())

    def test_analysis_removes_temp_file_after_success(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "truthlens_test.wav"
                path.write_bytes(b"test audio")

                upload = UploadFile(
                    filename="test.wav",
                    file=io.BytesIO(b"test audio"),
                )

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=path,
                    ),
                    patch.dict(api_server.detectors, {"audio": object()}),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                        return_value={"verdict": "Likely real"},
                    ),
                ):
                    result = await api_server.analyze_upload(upload, "audio")

                self.assertTrue(result["success"])
                self.assertFalse(path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())

    def test_analysis_removes_temp_file_after_detector_failure(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "truthlens_test.wav"
                path.write_bytes(b"test audio")

                upload = UploadFile(
                    filename="test.wav",
                    file=io.BytesIO(b"test audio"),
                )

                with (
                    patch.object(
                        api_server,
                        "save_upload",
                        new_callable=AsyncMock,
                        return_value=path,
                    ),
                    patch.dict(api_server.detectors, {"audio": object()}),
                    patch.object(
                        api_server,
                        "run_in_threadpool",
                        new_callable=AsyncMock,
                        side_effect=RuntimeError("Simulated detector failure"),
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.analyze_upload(upload, "audio")

                self.assertEqual(context.exception.status_code, 500)
                self.assertFalse(path.exists())
                self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()


    def test_audio_preview_rejects_unsupported_extension(self):
        async def run_test():
            upload = UploadFile(
                filename="recording.xyz",
                file=io.BytesIO(b"not audio"),
            )
            with self.assertRaises(HTTPException) as context:
                await api_server.convert_audio_preview(
                    api_server.BackgroundTasks(),
                    upload,
                )
            self.assertEqual(context.exception.status_code, 415)
            self.assertIn("Unsupported audio file format", context.exception.detail)

        asyncio.run(run_test())


    def test_audio_preview_rejects_empty_upload_and_closes_file(self):
        async def run_test():
            upload = UploadFile(
                filename="empty.wav",
                file=io.BytesIO(b""),
            )
            with self.assertRaises(HTTPException) as context:
                await api_server.convert_audio_preview(
                    api_server.BackgroundTasks(),
                    upload,
                )
            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("empty", context.exception.detail.lower())
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


    def test_audio_preview_returns_503_when_ffmpeg_is_unavailable(self):
        async def run_test():
            upload = UploadFile(
                filename="recording.wav",
                file=io.BytesIO(b"audio bytes"),
            )
            with patch.object(api_server.shutil, "which", return_value=None):
                with self.assertRaises(HTTPException) as context:
                    await api_server.convert_audio_preview(
                        api_server.BackgroundTasks(),
                        upload,
                    )
            self.assertEqual(context.exception.status_code, 503)
            self.assertIn("FFmpeg is not available", context.exception.detail)
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


    def test_audio_preview_rejects_upload_over_size_limit(self):
        async def run_test():
            upload = UploadFile(
                filename="large.wav",
                file=io.BytesIO(b"12345"),
            )
            with patch.object(api_server, "MAX_UPLOAD_BYTES", 4):
                with patch.object(api_server.shutil, "which") as ffmpeg_lookup:
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(
                            api_server.BackgroundTasks(),
                            upload,
                        )
            self.assertEqual(context.exception.status_code, 413)
            self.assertIn("100 MB upload limit", context.exception.detail)
            ffmpeg_lookup.assert_not_called()
            self.assertTrue(upload.file.closed)

        asyncio.run(run_test())


    def test_audio_preview_ffmpeg_failure_cleans_temporary_files(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                failed_ffmpeg = type(
                    "FailedProcess",
                    (),
                    {"returncode": 1, "stderr": b"simulated ffmpeg failure"},
                )()

                upload = UploadFile(
                    filename="broken.wav",
                    file=io.BytesIO(b"audio bytes"),
                )
                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ),
                    patch.object(
                        api_server.subprocess,
                        "run",
                        return_value=failed_ffmpeg,
                    ),
                ):
                    with self.assertRaises(HTTPException) as context:
                        await api_server.convert_audio_preview(
                            api_server.BackgroundTasks(),
                            upload,
                        )

                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("Audio preview conversion failed", context.exception.detail)
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())
