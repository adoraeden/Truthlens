import io
import math
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src import api_server
from src.image_detector import ImageDetector, LOGIT_THRESHOLD
from src.audio_detector import AudioDetector
from src.video_detector import VideoDetector


class GradientToyModel(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def preprocess_input(self, image):
        return torch.ones((1, 3, 384, 384), dtype=torch.float32)

    def vit(self, tensor):
        # Fixed boundary-valued output with a non-zero input gradient.
        value = tensor[:, :1, :1, :1] - tensor[:, :1, :1, :1].detach()
        return value.sum(dim=(1, 2, 3), keepdim=True) + self.logit


class ImageDetectorTests(unittest.TestCase):
    def make_detector(self, logit=LOGIT_THRESHOLD):
        detector = object.__new__(ImageDetector)
        detector.model = GradientToyModel(logit)
        return detector

    def test_score_and_verdict_share_exact_boundary_and_attribution_is_real(self):
        result = self.make_detector().predict(Image.new("RGB", (500, 400), "white"))
        self.assertEqual(result["logit"], LOGIT_THRESHOLD)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["verdict"], "AI-generated")
        self.assertEqual(result["explanation"]["method"], "positive input-gradient × input")
        self.assertEqual(result["explanation"]["scope"], "384 × 384 resized center crop used for inference")
        self.assertGreater(len(result["explanation"]["overlay_png_base64"]), 100)
        overlay = Image.open(io.BytesIO(__import__("base64").b64decode(result["explanation"]["overlay_png_base64"])))
        crop = Image.open(io.BytesIO(__import__("base64").b64decode(result["explanation"]["crop_png_base64"])))
        self.assertEqual(overlay.size, (384, 384))
        self.assertNotEqual(overlay.getpixel((0, 0)), crop.getpixel((0, 0)))
        self.assertEqual(result["image_size"], (500, 400))

    def test_documented_web_view_limits_long_edge_without_upscaling(self):
        large = Image.new("RGB", (4080, 3072), "white")
        view = ImageDetector._prepare_web_view(large)
        self.assertEqual(view.size, (768, 578))
        self.assertEqual(large.size, (4080, 3072))

        small = Image.new("RGB", (500, 400), "white")
        small_view = ImageDetector._prepare_web_view(small)
        self.assertEqual(small_view.size, (500, 400))

    def test_just_below_boundary_is_below_half_and_not_flagged(self):
        below_boundary = torch.nextafter(
            torch.tensor(LOGIT_THRESHOLD, dtype=torch.float32),
            torch.tensor(-math.inf, dtype=torch.float32),
        ).item()
        result = self.make_detector(below_boundary).predict(
            Image.new("RGB", (500, 400), "white")
        )
        self.assertLess(result["score"], 0.5)
        self.assertEqual(result["verdict"], "Likely real")

    def test_invalid_image_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as file:
            file.write(b"not an image")
            file.flush()
            with self.assertRaisesRegex(ValueError, "not a readable"):
                self.make_detector().predict(file.name)


class VideoDetectorTests(unittest.TestCase):
    def test_uniform_midpoint_sampling_caps_at_eight_and_keeps_modalities_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            video.touch()

            class Images:
                def predict(self, image):
                    return {"verdict": "Likely real", "score": 0.2, "logit": -1.0}

            detector = VideoDetector(Images(), SimpleNamespace(), max_frames=8)
            detector._probe_media = lambda _: (100.0, False)

            def extract(_, timestamp, output):
                Image.new("RGB", (20, 20)).save(output)
                return True

            detector._extract_frame = extract
            result = detector.predict(video)
            self.assertEqual(result["frames_requested"], 8)
            self.assertEqual(result["frames_analyzed"], 8)
            self.assertEqual(result["frame_timestamps_seconds"], [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75])
            self.assertEqual(result["audio_status"], "no_audio_stream")
            self.assertIsNone(result["audio_analysis"])
            self.assertIn("not a whole-video probability", result["frame_score_note"])

    def test_missing_video_is_rejected(self):
        detector = object.__new__(VideoDetector)
        with self.assertRaisesRegex(ValueError, "could not be found"):
            detector.predict("does-not-exist.mp4")

    def test_video_passes_precropped_audio_and_original_timing_to_audio_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            video.touch()
            calls = []

            class Images:
                def predict(self, image):
                    return {"verdict": "Likely real", "score": 0.2, "logit": -1.0}

            class Audio:
                def predict(self, path, **kwargs):
                    calls.append((path, kwargs))
                    return {
                        "source_duration_seconds": kwargs["source_duration_seconds"],
                        "analysis_start_seconds": kwargs["analysis_start_seconds"],
                        "analysis_region": "center_crop",
                    }

            detector = VideoDetector(Images(), Audio(), max_frames=1)
            detector._probe_media = lambda _: (13.123378, True)

            def extract_frame(_, timestamp, output):
                Image.new("RGB", (20, 20)).save(output)
                return True

            def extract_audio(_, output, duration):
                Path(output).write_bytes(b"cropped wav")
                return True, max(0.0, (duration - 4.0) / 2)

            detector._extract_frame = extract_frame
            detector._extract_audio = extract_audio
            result = detector.predict(video)

            self.assertEqual(len(calls), 1)
            _, kwargs = calls[0]
            self.assertTrue(kwargs["already_cropped"])
            self.assertEqual(kwargs["source_duration_seconds"], 13.123378)
            self.assertAlmostEqual(kwargs["analysis_start_seconds"], 4.561689)
            self.assertEqual(result["audio_analysis"]["source_duration_seconds"], 13.123378)
            self.assertAlmostEqual(result["audio_analysis"]["analysis_start_seconds"], 4.561689)
            self.assertEqual(result["audio_analysis"]["analysis_region"], "center_crop")


class AudioDetectorTests(unittest.TestCase):
    def test_unsupported_audio_extension_is_rejected_before_processing(self):
        detector = object.__new__(AudioDetector)
        with self.assertRaisesRegex(ValueError, "Unsupported audio format"):
            detector.predict("clip.xyz")

    def test_precropped_window_skips_seek_but_standalone_audio_keeps_center_crop(self):
        import src.audio_detector as audio_module

        detector = object.__new__(AudioDetector)
        detector.ffmpeg = "ffmpeg"
        detector.ffprobe = "ffprobe"
        detector.device = torch.device("cpu")

        class Model(torch.nn.Module):
            def forward(self, audio):
                return torch.zeros(1)

        detector.model = Model()
        ffmpeg_commands = []

        def fake_run(command, **kwargs):
            ffmpeg_commands.append(command)
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as folder:
            precropped = Path(folder) / "video_window.wav"
            standalone = Path(folder) / "standalone.m4a"
            precropped.write_bytes(b"input wav")
            standalone.write_bytes(b"input m4a")
            with (
                patch.object(detector, "_get_duration", side_effect=[4.0, 12.0]),
                patch.object(audio_module.subprocess, "run", side_effect=fake_run),
                patch.object(audio_module.pella, "load_clip", return_value=np.ones(64000, dtype=np.float32)),
            ):
                video_result = detector.predict(
                    precropped,
                    source_duration_seconds=13.123378,
                    analysis_start_seconds=4.561689,
                    already_cropped=True,
                )
                standalone_result = detector.predict(standalone)

        self.assertEqual(len(ffmpeg_commands), 2)
        self.assertNotIn("-ss", ffmpeg_commands[0])
        self.assertEqual(video_result["duration_analyzed_seconds"], 4.0)
        self.assertEqual(video_result["source_duration_seconds"], 13.12)
        self.assertEqual(video_result["analysis_region"], "center_crop")
        self.assertEqual(video_result["analysis_start_seconds"], 4.56)

        standalone_seek = ffmpeg_commands[1].index("-ss")
        self.assertEqual(ffmpeg_commands[1][standalone_seek + 1], "4.000000")
        self.assertEqual(standalone_result["source_duration_seconds"], 12.0)
        self.assertEqual(standalone_result["analysis_region"], "center_crop")
        self.assertEqual(standalone_result["analysis_start_seconds"], 4.0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_silent_wav_is_rejected_after_16khz_conversion(self):
        import src.audio_detector as audio_module

        detector = object.__new__(AudioDetector)
        detector.ffmpeg = shutil.which("ffmpeg")
        detector.ffprobe = shutil.which("ffprobe")
        detector.device = torch.device("cpu")
        detector.model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "silent.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 8000)
            with patch.object(audio_module.pella, "load_clip", return_value=np.zeros(64000, dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, "recording is silent"):
                    detector.predict(audio)


class ApiCompatibilityTests(unittest.TestCase):
    def test_existing_multipart_routes_remain_declared(self):
        from fastapi.routing import APIRoute

        routes = {
            route.path: route
            for route in api_server.app.routes
            if isinstance(route, APIRoute)
        }
        self.assertTrue({"/api/detect/image", "/api/detect/audio", "/api/detect/video"}.issubset(routes))
        for path in ("/api/detect/image", "/api/detect/audio", "/api/detect/video"):
            self.assertIn("POST", routes[path].methods)


if __name__ == "__main__":
    unittest.main()
