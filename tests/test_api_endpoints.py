import io
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
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
                files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 415)
            self.assertIn("Unsupported video file format", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
