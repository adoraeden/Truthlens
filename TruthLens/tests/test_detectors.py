import asyncio
import io
import json
import math
import os
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
from src import image_detector as image_detector_module
from src.image_detector import ImageDetector, LOGIT_THRESHOLD
from src.image_diagnostics import (
    build_visualizations,
    collect_source_metadata,
    compute_attribution_heatmap,
    compute_ela,
    compute_noise_residual,
    compute_rgb_analysis,
)
from src.audio_detector import AudioDetector
from src.video_detector import VideoDetector
from eval_image_thresholds import metrics as image_threshold_metrics
from eval_audio_conditions import classification_metrics as audio_metrics, require_pella_memory
from eval_video_methods import (
    classification_metrics as video_metrics,
    require_image_memory,
    summarize_methods,
)
from src.image_calibration import calibrated_probabilities, load_calibrator
from calibrate_image_detector import audit_splits, fit_calibrator, metrics as calibration_metrics, probabilities


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

    def test_image_threshold_defaults_to_zero_and_supports_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(image_detector_module._configured_image_logit_threshold(), 0.0)
        with patch.dict(os.environ, {"TRUTHLENS_IMAGE_LOGIT_THRESHOLD": "2.0"}, clear=True):
            self.assertEqual(image_detector_module._configured_image_logit_threshold(), 2.0)
        with patch.dict(os.environ, {"TRUTHLENS_IMAGE_LOGIT_THRESHOLD": "NaN"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "finite number"):
                image_detector_module._configured_image_logit_threshold()

    def test_default_threshold_is_zero_and_predictions_switch_at_boundary(self):
        self.assertEqual(LOGIT_THRESHOLD, 0.0)
        below = -1e-5
        above = 1e-5
        below_result = self.make_detector(below).predict(Image.new("RGB", (32, 24), "white"))
        above_result = self.make_detector(above).predict(Image.new("RGB", (32, 24), "white"))
        self.assertEqual(below_result["verdict"], "Likely real")
        self.assertLess(below_result["score"], 0.5)
        self.assertEqual(above_result["verdict"], "AI-generated")
        self.assertGreater(above_result["score"], 0.5)
        self.assertEqual(below_result["decision_threshold"], 0.0)
        self.assertEqual(above_result["decision_threshold"], 0.0)

    def test_score_and_verdict_share_exact_boundary_and_attribution_is_real(self):
        result = self.make_detector().predict(Image.new("RGB", (500, 400), "white"))
        self.assertEqual(result["logit"], LOGIT_THRESHOLD)
        self.assertEqual(result["positive_class"], "AI-generated")
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["verdict"], "AI-generated")
        self.assertEqual(result["decision_threshold"], LOGIT_THRESHOLD)
        self.assertEqual(result["score_cutoff"], 0.5)
        self.assertEqual(result["score_method"], "sigmoid(logit - decision_threshold)")
        self.assertFalse(result["score_calibrated_probability"])
        self.assertEqual(result["explanation"]["method"], "positive input-gradient × input")
        self.assertEqual(result["explanation"]["scope"], "384 × 384 resized center crop used for inference")
        self.assertGreater(len(result["explanation"]["overlay_png_base64"]), 100)
        overlay = Image.open(io.BytesIO(__import__("base64").b64decode(result["explanation"]["overlay_png_base64"])))
        crop = Image.open(io.BytesIO(__import__("base64").b64decode(result["explanation"]["crop_png_base64"])))
        heatmap = Image.open(io.BytesIO(__import__("base64").b64decode(result["explanation"]["heatmap_png_base64"])))
        self.assertEqual(overlay.size, (384, 384))
        self.assertEqual(heatmap.size, crop.size)
        self.assertNotEqual(overlay.getpixel((0, 0)), crop.getpixel((0, 0)))
        self.assertEqual(result["explanation"]["spatial_attribution"]["grid_size"], 4)
        self.assertEqual(len(result["explanation"]["spatial_attribution"]["cells"]), 16)
        self.assertEqual(result["explanation"]["rgb"]["pixel_count"], 500 * 400)
        self.assertIn('"heatmap_png_base64"', json.dumps(result))
        self.assertEqual(result["image_size"], (500, 400))
        self.assertEqual(result["calibration_status"], "unavailable")
        self.assertIsNone(result["ai_generated_probability"])
        self.assertIsNone(result["genuine_probability"])

    def test_attribution_heatmap_handles_zero_and_constant_maps(self):
        zero_map, zero_regions = compute_attribution_heatmap(np.zeros((384, 384), dtype=np.float32))
        self.assertEqual(zero_map.size, (384, 384))
        self.assertEqual(zero_regions["distribution"], "no_positive_attribution")
        self.assertEqual(sum(cell["percentage_of_positive_total"] for cell in zero_regions["cells"]), 0)

        constant_map, constant_regions = compute_attribution_heatmap(np.ones((384, 384), dtype=np.float32))
        self.assertEqual(constant_map.size, (384, 384))
        self.assertEqual(constant_regions["distribution"], "distributed")
        for cell in constant_regions["cells"]:
            self.assertAlmostEqual(cell["percentage_of_positive_total"], 6.25, places=5)

    def test_ela_and_noise_maps_are_real_pixel_calculations(self):
        gray = Image.new("L", (64, 64), 127)
        ela_gray, ela_scale = compute_ela(gray)
        noise_gray, noise_scale = compute_noise_residual(gray)
        self.assertEqual(ela_gray.size, (64, 64))
        self.assertEqual(noise_gray.size, (64, 64))
        self.assertEqual(noise_scale, 0)
        self.assertEqual(noise_gray.getextrema(), ((0, 0), (0, 0), (0, 0)))

        pattern = Image.new("RGB", (64, 64))
        pattern.putdata([
            (x * 4 % 256, y * 4 % 256, (x * 7 + y * 11) % 256)
            for y in range(64) for x in range(64)
        ])
        ela_map, ela_scale = compute_ela(pattern, quality=60)
        residual_map, residual_scale = compute_noise_residual(pattern)
        self.assertEqual(ela_map.size, pattern.size)
        self.assertGreater(ela_scale, 0)
        self.assertEqual(residual_map.size, pattern.size)
        self.assertGreater(residual_scale, 0)
        self.assertGreater(max(max(channel) for channel in ela_map.getextrema()), 0)
        self.assertGreater(max(max(channel) for channel in residual_map.getextrema()), 0)

    def test_rgb_histograms_and_statistics_use_exact_channel_pixels(self):
        image = Image.new("RGB", (2, 2))
        image.putdata([(0, 10, 20), (100, 10, 40), (200, 30, 40), (200, 50, 60)])
        analysis = compute_rgb_analysis(image)
        self.assertEqual(analysis["pixel_count"], 4)
        self.assertEqual(analysis["histograms"]["red"][200], 2)
        self.assertEqual(analysis["histograms"]["green"][10], 2)
        self.assertEqual(analysis["statistics"]["red"]["minimum"], 0)
        self.assertEqual(analysis["statistics"]["red"]["maximum"], 200)
        self.assertAlmostEqual(analysis["statistics"]["red"]["mean"], 125.0)

    def test_grayscale_and_rgba_inputs_are_supported_and_diagnostics_do_not_change_verdict(self):
        detector = self.make_detector(logit=2.5)
        grayscale = detector.predict(Image.new("L", (32, 24), 128))
        rgba = detector.predict(Image.new("RGBA", (32, 24), (20, 40, 60, 100)))
        self.assertEqual(grayscale["verdict"], "AI-generated")
        self.assertEqual(rgba["verdict"], "AI-generated")
        self.assertEqual(grayscale["score"], rgba["score"])

        with patch("src.image_detector.build_visualizations", return_value={}) as diagnostics:
            without_diagnostics = detector.predict(Image.new("RGB", (32, 24), "white"))
        self.assertEqual(without_diagnostics["verdict"], "AI-generated")
        self.assertEqual(without_diagnostics["logit"], 2.5)
        diagnostics.assert_called_once()

    def test_compression_metadata_only_reports_fields_present_in_source(self):
        with tempfile.TemporaryDirectory() as folder:
            png_path = Path(folder) / "source.png"
            jpeg_path = Path(folder) / "source.jpg"
            source = Image.new("RGB", (24, 16), "navy")
            source.save(png_path, format="PNG")
            source.save(jpeg_path, format="JPEG", quality=73)

            for path, expected_format in ((png_path, "PNG"), (jpeg_path, "JPEG")):
                with Image.open(path) as opened:
                    metadata = collect_source_metadata(opened)
                    opened.load()
                    source_copy = opened.convert("RGB")
                diagnostic = build_visualizations(
                    source_copy,
                    np.zeros((384, 384), dtype=np.float32),
                    metadata,
                )
                compression = diagnostic["compression"]
                self.assertEqual(compression["source_format"], expected_format)
                self.assertIsNone(compression["quality_estimate"])
                if expected_format == "PNG":
                    self.assertIsNone(compression["jpeg"])
                    self.assertIn("not JPEG", compression["note"])
                else:
                    self.assertIsNotNone(compression["jpeg"])
                    self.assertGreater(len(compression["jpeg"]["quantization_tables"]), 0)


class ImageCalibrationTests(unittest.TestCase):
    def test_missing_and_ineligible_calibrators_are_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.json"
            self.assertIsNone(load_calibrator(missing))
            candidate = Path(folder) / "candidate.json"
            candidate.write_text(json.dumps({"schema_version": 1, "production_eligible": False}), encoding="utf-8")
            self.assertIsNone(load_calibrator(candidate))

    def test_loader_accepts_only_valid_eligible_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibrator.json"
            artifact = {
                "schema_version": 1, "production_eligible": True, "dataset_version": "grouped-v1",
                "decision_threshold": LOGIT_THRESHOLD, "method": "temperature", "parameters": {"temperature": 1.4},
            }
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(load_calibrator(path)["method"], "temperature")
            artifact["decision_threshold"] = LOGIT_THRESHOLD + 1
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertIsNone(load_calibrator(path))

    def test_valid_calibrator_probabilities_are_bounded_and_complementary(self):
        artifact = {"method": "platt", "parameters": {"slope": 0.7, "intercept": -0.4}}
        result = calibrated_probabilities(2.0, artifact)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["ai_generated_probability"], 0)
        self.assertLessEqual(result["ai_generated_probability"], 1)
        self.assertAlmostEqual(result["ai_generated_probability"] + result["genuine_probability"], 1)
        self.assertIsNone(calibrated_probabilities(float("nan"), artifact))

    def test_calibration_methods_fit_and_metrics_are_computed(self):
        logits = [-2.5, -1.8, -1.2, 0.2, 0.8, 1.5, 2.1, 3.0]
        truths = [0, 0, 0, 0, 1, 1, 1, 1]
        for method in ("platt", "temperature"):
            params = fit_calibrator(method, logits, truths)
            preds = probabilities(method, params, logits)
            self.assertTrue(np.all((preds >= 0) & (preds <= 1)))
            report = calibration_metrics(truths, preds)
            self.assertIn("brier_score", report)
            self.assertIn("negative_log_likelihood", report)
            self.assertIn("expected_calibration_error_10_bins", report)
            self.assertIn("roc_auc", report)
            self.assertIn("pr_auc_average_precision", report)
            self.assertIn("genuine", report["by_class"])
            self.assertIn("ai_generated", report["by_class"])

    def test_group_and_duplicate_overlap_aborts_split_audit(self):
        base = {"truth": 0, "logit": -1.0, "sha256": "hash-a", "source_id": "source-a", "generator_id": "gen-a", "original_id": "original-a"}
        with self.assertRaisesRegex(ValueError, "source_id overlap"):
            audit_splits({"calibration": [base], "development": [{**base, "sha256": "hash-b"}]})

    def test_probability_layer_does_not_change_raw_verdict(self):
        detector = object.__new__(ImageDetector)
        detector.model = GradientToyModel(LOGIT_THRESHOLD)
        detector.calibrator = {"method": "platt", "parameters": {"slope": 0.1, "intercept": -4.0}}
        result = detector.predict(Image.new("RGB", (32, 24), "white"))
        self.assertEqual(result["verdict"], "AI-generated")
        self.assertEqual(result["logit"], LOGIT_THRESHOLD)
        self.assertLess(result["ai_generated_probability"], 0.5)

class ImageDetectorLimitTests(unittest.TestCase):
    def make_detector(self, logit=LOGIT_THRESHOLD):
        detector = object.__new__(ImageDetector)
        detector.model = GradientToyModel(logit)
        return detector

    def test_image_pixel_and_byte_limits_are_enforced(self):
        import stat
        import src.image_detector as image_module

        detector = self.make_detector()
        with patch.object(image_module, "MAX_IMAGE_PIXELS", 10):
            with self.assertRaisesRegex(ValueError, "100-megapixel"):
                detector.predict(Image.new("RGB", (4, 4), "white"))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "valid.png"
            Image.new("RGB", (4, 4), "white").save(path)
            real_stat = Path.stat

            def oversized_stat(target, *args, **kwargs):
                if target == path:
                    return type("FileStat", (), {
                        "st_size": image_module.MAX_IMAGE_BYTES + 1,
                        "st_mode": stat.S_IFREG | 0o600,
                    })()
                return real_stat(target, *args, **kwargs)

            with patch.object(Path, "stat", oversized_stat):
                with self.assertRaisesRegex(ValueError, "100 MB"):
                    detector.predict(path)

    def test_score_formula_matches_reported_example_logit_and_threshold(self):
        result = self.make_detector(logit=0.386).predict(Image.new("RGB", (500, 400), "white"))
        expected = 1 / (1 + math.exp(-(0.386 - LOGIT_THRESHOLD)))
        self.assertAlmostEqual(result["score"], expected, places=7)
        self.assertEqual(result["verdict"], "AI-generated" if 0.386 >= LOGIT_THRESHOLD else "Likely real")

    def test_nonfinite_model_logit_is_rejected_without_a_verdict(self):
        with self.assertRaisesRegex(ValueError, "non-finite logit"):
            self.make_detector(float("nan")).predict(Image.new("RGB", (500, 400), "white"))

    def test_saved_image_threshold_metrics_keep_ai_as_positive_class(self):
        rows = [(1, 2.2), (1, 1.8), (0, 2.1), (0, 1.7)]
        result = image_threshold_metrics(rows, 2.0)
        self.assertEqual((result["tp"], result["fn"], result["fp"], result["tn"]), (1, 1, 1, 1))

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
        self.assertLess(result["logit"], LOGIT_THRESHOLD)
        self.assertAlmostEqual(result["score"], 0.5, places=6)
        self.assertEqual(result["verdict"], "Likely real")

    def test_invalid_image_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as file:
            file.write(b"not an image")
            file.flush()
            with self.assertRaisesRegex(ValueError, "not a readable"):
                self.make_detector().predict(file.name)

    def test_exif_orientation_is_applied_without_changing_reported_source_dimensions(self):
        detector = self.make_detector()
        prepared_sizes = []
        detector._prepare_web_view = lambda image: prepared_sizes.append(image.size) or image
        exif = Image.Exif()
        exif[274] = 6

        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "oriented.jpg"
            Image.new("RGB", (20, 10), "white").save(image_path, exif=exif)
            result = detector.predict(image_path)

        self.assertEqual(prepared_sizes, [(10, 20)])
        self.assertEqual(result["image_size"], (20, 10))


class VideoDetectorTests(unittest.TestCase):
    def test_video_keeps_visual_result_when_audio_model_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            video.touch()

            class Images:
                def predict(self, image):
                    return {"verdict": "Likely real", "score": 0.2, "logit": -1.0}

            detector = VideoDetector(
                Images(),
                None,
                max_frames=1,
                audio_unavailable_reason="PellaV2 skipped by RAM guard.",
            )
            detector._probe_media = lambda _: (4.0, True)
            detector._extract_frame = lambda _, __, output: Image.new("RGB", (20, 20)).save(output) is None
            detector._extract_audio = lambda *_: self.fail("Audio should not be decoded without an audio detector")
            result = detector.predict(video)

        self.assertEqual(result["frames_analyzed"], 1)
        self.assertEqual(result["audio_status"], "audio_detector_unavailable")
        self.assertEqual(result["audio_status_note"], "PellaV2 skipped by RAM guard.")
        self.assertIsNone(result["audio_analysis"])

    def test_video_keeps_visual_result_when_audio_inference_raises_unexpected_error(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            video.touch()

            class Images:
                def predict(self, image):
                    return {"verdict": "Likely real", "score": 0.2, "logit": -1.0}

            class Audio:
                def predict(self, *args, **kwargs):
                    raise RuntimeError("Unexpected audio inference failure")

            detector = VideoDetector(Images(), Audio(), max_frames=1)
            detector._probe_media = lambda _: (4.0, True)
            detector._extract_frame = lambda _, __, output: Image.new("RGB", (20, 20)).save(output) is None
            detector._extract_audio = lambda _, output, duration: (True, 0.0)

            result = detector.predict(video)

        self.assertEqual(result["frames_analyzed"], 1)
        self.assertEqual(result["audio_status"], "audio_stream_unreadable")
        self.assertIsNone(result["audio_analysis"])
        self.assertIn("could not be decoded", result["audio_status_note"])


    def test_video_method_comparison_is_robust_to_one_flagged_outlier(self):
        summary = summarize_methods([0.9, 0.1, 0.1, 0.1, 0.1])
        self.assertAlmostEqual(summary["mean"], 0.26)
        self.assertAlmostEqual(summary["median"], 0.1)
        self.assertAlmostEqual(summary["flagged_fraction"], 0.2)
        self.assertEqual(
            video_metrics([1, 0], [1, 0]),
            {"n": 2, "tp": 1, "fp": 0, "tn": 1, "fn": 0,
             "precision": 1.0, "recall": 1.0, "f1": 1.0,
             "fpr": 0.0, "fnr": 0.0, "accuracy": 1.0},
        )

    def test_model_evaluators_refuse_to_start_below_documented_memory_guards(self):
        with self.assertRaisesRegex(RuntimeError, r"at least \d+\.\d+ GB"):
            require_pella_memory(0.7)
        with self.assertRaisesRegex(RuntimeError, "at least 1.5 GB"):
            require_image_memory(0.7)
        self.assertEqual(audio_metrics([1, 0, 1], [0, 1, 1]), {
            "n": 3, "tp": 1, "fp": 1, "tn": 0, "fn": 1,
            "accuracy": 1 / 3, "precision": 0.5, "recall": 0.5,
            "f1": 0.5, "fpr": 1.0, "fnr": 0.5,
        })

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

    def test_video_uses_bounded_two_second_samples_and_robust_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            video.touch()
            timestamps = []
            frame_outputs = iter([
                {"verdict": "AI-generated", "score": 0.9, "logit": 3.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
                {"verdict": "Likely real", "score": 0.1, "logit": -2.0},
            ])

            class Images:
                def predict(self, image):
                    return next(frame_outputs)

            detector = VideoDetector(Images(), SimpleNamespace(), max_frames=8)
            detector._probe_media = lambda _: (13.0, False)

            def extract(_, timestamp, output):
                timestamps.append(timestamp)
                Image.new("RGB", (20, 20)).save(output)
                return True

            detector._extract_frame = extract
            result = detector.predict(video)

        self.assertEqual(result["frames_requested"], 7)
        self.assertEqual(len(timestamps), 7)
        self.assertAlmostEqual(timestamps[0], 13 / 14)
        self.assertAlmostEqual(timestamps[-1], 13 * 13 / 14)
        self.assertEqual(result["ai_flagged_frames"], 1)
        self.assertAlmostEqual(result["average_frame_score"], (0.9 + 6 * 0.1) / 7)
        self.assertEqual(result["median_frame_score"], 0.1)
        self.assertAlmostEqual(result["flagged_frame_fraction"], 1 / 7)
        self.assertEqual(result["robust_frame_summary"], "median_frame_score")
        self.assertNotIn("verdict", result)
        self.assertIn("not a whole-video probability", result["frame_score_note"])
        self.assertIn("not a whole-video authenticity verdict", result["robust_frame_summary_note"])
        self.assertIn("do not establish the authenticity", result["interpretation"])
        self.assertEqual(result["audio_status"], "no_audio_stream")

    def test_missing_video_is_rejected(self):
        detector = object.__new__(VideoDetector)
        with self.assertRaisesRegex(ValueError, "could not be found"):
            detector.predict("does-not-exist.mp4")

    def test_probe_media_rejects_missing_video_stream_and_invalid_duration(self):
        import src.video_detector as video_module

        detector = object.__new__(VideoDetector)
        detector.ffprobe = "ffprobe"
        invalid_cases = [
            (
                {"streams": [{"codec_type": "audio"}], "format": {"duration": "2.0"}},
                "does not contain a video stream",
            ),
            (
                {
                    "streams": [{"codec_type": "video", "duration": "nan"}],
                    "format": {"duration": "inf"},
                },
                "duration could not be determined",
            ),
        ]

        for metadata, message in invalid_cases:
            with self.subTest(metadata=metadata):
                completed = SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(metadata),
                    stderr="",
                )
                with patch.object(video_module.subprocess, "run", return_value=completed):
                    with self.assertRaisesRegex(ValueError, message):
                        detector._probe_media("clip.mp4")

    def test_probe_media_rejects_malformed_json(self):
        import src.video_detector as video_module

        detector = object.__new__(VideoDetector)
        detector.ffprobe = "ffprobe"
        completed = SimpleNamespace(returncode=0, stdout="{not-json", stderr="")

        with patch.object(video_module.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "invalid or cannot be read"):
                detector._probe_media("clip.mp4")

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
    def test_audio_detector_memory_guard_runs_before_model_construction(self):
        import src.audio_detector as audio_module

        with (
            patch.object(audio_module, "available_memory_gb", return_value=0.7),
            patch.dict("sys.modules", {"pellav2_infer": SimpleNamespace(Detector=lambda: self.fail("model must not load"))}),
        ):
            with self.assertRaisesRegex(RuntimeError, r"at least \d+\.\d+ GB"):
                AudioDetector()

    def test_unsupported_audio_extension_is_rejected_before_processing(self):
        detector = object.__new__(AudioDetector)
        with self.assertRaisesRegex(ValueError, "Unsupported audio format"):
            detector.predict("clip.xyz")

    def test_audio_duration_rejects_nonfinite_zero_and_too_short_values(self):
        import src.audio_detector as audio_module

        detector = object.__new__(AudioDetector)
        detector.ffprobe = "ffprobe"

        for output in ("nan", "inf", "0", "0.1"):
            with self.subTest(duration=output):
                completed = SimpleNamespace(returncode=0, stdout=output)
                with patch.object(audio_module.subprocess, "run", return_value=completed):
                    with self.assertRaisesRegex(ValueError, "invalid|too short"):
                        detector._get_duration("clip.wav")

    def test_nonfinite_audio_waveform_is_rejected(self):
        import src.audio_detector as audio_module

        detector = object.__new__(AudioDetector)
        detector.ffmpeg = "ffmpeg"
        detector.ffprobe = "ffprobe"
        detector.device = torch.device("cpu")
        detector.model = SimpleNamespace()

        def fake_ffmpeg(command, **kwargs):
            Path(command[-1]).write_bytes(b"converted wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "invalid.wav"
            audio.write_bytes(b"input audio")
            with (
                patch.object(detector, "_get_duration", return_value=1.0),
                patch.object(audio_module.subprocess, "run", side_effect=fake_ffmpeg),
                patch.object(
                    detector,
                    "_pella",
                    SimpleNamespace(
                        load_clip=lambda _: np.array([np.nan], dtype=np.float32)
                    ),
                    create=True,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "empty or invalid"):
                    detector.predict(audio)

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
                patch.object(detector, "_pella", SimpleNamespace(load_clip=lambda _: np.ones(64000, dtype=np.float32)), create=True),
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
            with patch.object(detector, "_pella", SimpleNamespace(load_clip=lambda _: np.zeros(64000, dtype=np.float32)), create=True):
                with self.assertRaisesRegex(ValueError, "recording is silent"):
                    detector.predict(audio)


class ApiCompatibilityTests(unittest.TestCase):
    def test_image_api_payload_uses_active_threshold_for_score_and_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "api.png"
            Image.new("RGB", (32, 24), "white").save(image_path)
            previous = api_server.detectors["image"]
            detector = object.__new__(ImageDetector)
            detector.model = GradientToyModel(0.25)
            api_server.detectors["image"] = detector

            async def fake_save_upload(_upload, _modality):
                return image_path

            upload = api_server.UploadFile(filename="api.png", file=io.BytesIO(b"upload"))
            try:
                with patch.object(api_server, "save_upload", new=fake_save_upload):
                    response = asyncio.run(api_server.analyze_upload(upload, "image"))
            finally:
                api_server.detectors["image"] = previous

        result = response["result"]
        self.assertEqual(result["decision_threshold"], 0.0)
        self.assertEqual(result["logit"], 0.25)
        self.assertEqual(result["verdict"], "AI-generated")
        self.assertAlmostEqual(result["score"], 1 / (1 + math.exp(-0.25)))
        self.assertFalse(result["score_calibrated_probability"])

    def test_pella_ram_gate_and_degraded_health_keep_image_service_available(self):
        self.assertFalse(api_server.pella_can_load(0.7))
        self.assertFalse(api_server.pella_can_load(float("nan")))
        self.assertTrue(api_server.pella_can_load(5.0))

        previous = dict(api_server.detectors)
        try:
            api_server.detectors.update(image=object(), audio=None, video=object())
            result = api_server.health()
            self.assertEqual(result["status"], "online")
            self.assertTrue(result["detectors"]["image"])
            self.assertFalse(result["detectors"]["audio"])
        finally:
            api_server.detectors.update(previous)

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
