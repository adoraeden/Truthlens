"""Comprehensive functional audit script for TruthLens Audio Lab and Video Lab.
Tests complete inference pipelines, formats, edge cases, and API compatibility.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import math
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Set minimum RAM threshold to allow running in current environment (if RAM >= 2.0 GB)
os.environ["TRUTHLENS_PELLA_MIN_RAM_GB"] = "2.0"

import torch
import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "pellav2"))

from src.audio_detector import AudioDetector, SUPPORTED_AUDIO_EXTENSIONS, available_memory_gb
from src.image_detector import ImageDetector
from src.video_detector import VideoDetector, SUPPORTED_VIDEO_EXTENSIONS


def run_audit():
    print("=" * 70)
    print("STARTING TRUTHLENS AUDIO & VIDEO FUNCTIONAL AUDIT")
    print("=" * 70)

    audit_results = {
        "audio": {},
        "video": {},
        "api": {},
    }

    # -------------------------------------------------------------
    # 1. AUDIO DETECTOR AUDIT
    # -------------------------------------------------------------
    print("\n[1] AUDITING AUDIO DETECTION PIPELINE (PellaV2 + FFmpeg)")
    mem_gb = available_memory_gb()
    print(f"Available RAM: {mem_gb:.2f} GB")

    try:
        audio_detector = AudioDetector()
        print("[PASS] AudioDetector initialized successfully (PellaV2 + Wav2Vec-BERT loaded).")
        audit_results["audio"]["model_loaded"] = True
    except Exception as exc:
        print(f"[FAIL] AudioDetector initialization failed: {exc}")
        audit_results["audio"]["model_loaded"] = False
        audio_detector = None

    if audio_detector is not None:
        # Check actual real audio file
        real_wav = ROOT / "test_audio_wav" / "real1.wav"
        if real_wav.is_file():
            res = audio_detector.predict(real_wav)
            print(f"[PASS] Inference on real sample ({real_wav.name}):")
            print(f"  Verdict: {res['verdict']} | Score: {res['score']:.4f} | Analyzed: {res['duration_analyzed_seconds']}s")
            audit_results["audio"]["real_sample"] = res
            assert res["verdict"] == "Likely real", f"Expected Likely real, got {res['verdict']}"
            assert res["score"] < 0.5, f"Expected score < 0.5, got {res['score']}"
        else:
            print("[WARN] real1.wav not found")

        # Check actual fake audio file
        fake_wav = ROOT / "test_audio_wav" / "fake1.wav"
        if fake_wav.is_file():
            res = audio_detector.predict(fake_wav)
            print(f"[PASS] Inference on synthetic sample ({fake_wav.name}):")
            print(f"  Verdict: {res['verdict']} | Score: {res['score']:.4f} | Analyzed: {res['duration_analyzed_seconds']}s")
            audit_results["audio"]["fake_sample"] = res
            assert res["verdict"] == "AI-generated", f"Expected AI-generated, got {res['verdict']}"
            assert res["score"] >= 0.5, f"Expected score >= 0.5, got {res['score']}"
        else:
            print("[WARN] fake1.wav not found")

        # Test all supported formats: WAV, MP3, M4A, FLAC, OGG, AAC
        print("\n--- Testing All Supported Audio Formats ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a 6-second synthetic 440 Hz tone test audio
            sr = 44100
            t = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
            sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            base_wav = temp_path / "source_base.wav"
            sf.write(str(base_wav), sine_wave, sr)

            formats = ["wav", "mp3", "m4a", "flac", "ogg", "aac"]
            format_results = {}
            for fmt in formats:
                target_file = temp_path / f"test_tone.{fmt}"
                # Convert with ffmpeg
                cmd = [
                    shutil.which("ffmpeg"), "-y", "-v", "error",
                    "-i", str(base_wav),
                    "-c:a", "pcm_s16le" if fmt == "wav" else "aac" if fmt in ("m4a", "aac") else "libmp3lame" if fmt == "mp3" else "flac" if fmt == "flac" else "libvorbis",
                    str(target_file),
                ]
                subprocess.run(cmd, check=True)
                assert target_file.is_file() and target_file.stat().st_size > 0

                pred = audio_detector.predict(target_file)
                print(f"[PASS] Format .{fmt.upper()}: Verdict={pred['verdict']}, Score={pred['score']:.4f}, Duration={pred['source_duration_seconds']:.2f}s, Analyzed={pred['duration_analyzed_seconds']}s, Region={pred['analysis_region']}")
                format_results[fmt] = {
                    "success": True,
                    "verdict": pred["verdict"],
                    "score": pred["score"],
                    "analysis_region": pred["analysis_region"],
                }
                assert pred["analysis_region"] == "center_crop"
                assert math.isclose(pred["duration_analyzed_seconds"], 4.0, abs_tol=0.1)

            audit_results["audio"]["formats"] = format_results

        # Test 4-Second Center Crop Logic
        print("\n--- Testing 4-Second Center Crop Timing ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Test 10-second file: center crop should start at (10 - 4)/2 = 3.0s
            t_10 = np.linspace(0, 10.0, int(16000 * 10.0), endpoint=False)
            tone_10 = (0.3 * np.sin(2 * np.pi * 300 * t_10)).astype(np.float32)
            wav_10 = temp_path / "long_10s.wav"
            sf.write(str(wav_10), tone_10, 16000)

            pred_10 = audio_detector.predict(wav_10)
            print(f"[PASS] 10-second file: analysis_start_seconds={pred_10['analysis_start_seconds']}s (expected 3.00s)")
            assert math.isclose(pred_10["analysis_start_seconds"], 3.0, abs_tol=0.05)
            assert pred_10["analysis_region"] == "center_crop"

            # Test 2.5-second file (< 4s): full recording padded, analysis_start_seconds=0.0s
            t_25 = np.linspace(0, 2.5, int(16000 * 2.5), endpoint=False)
            tone_25 = (0.3 * np.sin(2 * np.pi * 300 * t_25)).astype(np.float32)
            wav_25 = temp_path / "short_2.5s.wav"
            sf.write(str(wav_25), tone_25, 16000)

            pred_25 = audio_detector.predict(wav_25)
            print(f"[PASS] 2.5-second file: analysis_start_seconds={pred_25['analysis_start_seconds']}s, region={pred_25['analysis_region']}")
            assert math.isclose(pred_25["analysis_start_seconds"], 0.0, abs_tol=0.05)
            assert pred_25["analysis_region"] == "full_recording"

        # Test Error Handling: Silent audio
        print("\n--- Testing Silent Audio Rejection ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            silent_path = Path(temp_dir) / "silent.wav"
            silent_data = np.zeros(16000 * 4, dtype=np.float32)
            sf.write(str(silent_path), silent_data, 16000)
            try:
                audio_detector.predict(silent_path)
                print("[FAIL] Silent audio was NOT rejected!")
                audit_results["audio"]["silent_rejected"] = False
            except ValueError as exc:
                print(f"[PASS] Silent audio correctly rejected: '{exc}'")
                audit_results["audio"]["silent_rejected"] = True

        # Test Error Handling: Too short audio (< 0.25s)
        print("\n--- Testing Too Short Audio Rejection (< 0.25s) ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            short_path = Path(temp_dir) / "too_short.wav"
            short_data = np.random.randn(int(16000 * 0.1)).astype(np.float32)
            sf.write(str(short_path), short_data, 16000)
            try:
                audio_detector.predict(short_path)
                print("[FAIL] Too short audio was NOT rejected!")
                audit_results["audio"]["too_short_rejected"] = False
            except ValueError as exc:
                print(f"[PASS] Too short audio correctly rejected: '{exc}'")
                audit_results["audio"]["too_short_rejected"] = True

        # Test Error Handling: Malformed / Corrupted file
        print("\n--- Testing Malformed Audio Rejection ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt_path = Path(temp_dir) / "corrupt.mp3"
            corrupt_path.write_bytes(b"NOT_A_VALID_AUDIO_FILE_DATA_CORRUPT")
            try:
                audio_detector.predict(corrupt_path)
                print("[FAIL] Malformed audio was NOT rejected!")
                audit_results["audio"]["corrupt_rejected"] = False
            except ValueError as exc:
                print(f"[PASS] Malformed audio correctly rejected: '{exc}'")
                audit_results["audio"]["corrupt_rejected"] = True

    # -------------------------------------------------------------
    # 2. VIDEO DETECTOR AUDIT
    # -------------------------------------------------------------
    print("\n[2] AUDITING VIDEO DETECTION PIPELINE (ViT + Midpoint Sampling + Audio)")
    try:
        image_detector = ImageDetector()
        print("[PASS] ImageDetector initialized successfully (ViT-Small/16 loaded).")
        video_detector = VideoDetector(
            image_detector=image_detector,
            audio_detector=audio_detector,
            max_frames=8,
        )
        print("[PASS] VideoDetector initialized successfully.")
        audit_results["video"]["initialized"] = True
    except Exception as exc:
        print(f"[FAIL] VideoDetector initialization failed: {exc}")
        audit_results["video"]["initialized"] = False
        video_detector = None

    if video_detector is not None:
        # Test on existing sample video Golgappe-Ka-SnapYT.App.mp4
        sample_video = ROOT / "test_videos" / "Golgappe-Ka-SnapYT.App.mp4"
        if sample_video.is_file():
            print(f"\n--- Testing Video Inference on {sample_video.name} ---")
            vid_result = video_detector.predict(sample_video)
            print(f"[PASS] Duration: {vid_result['duration_seconds']}s")
            print(f"[PASS] Frames Analyzed: {vid_result['frames_analyzed']} (Requested: {vid_result['frames_requested']})")
            print(f"[PASS] Flagged Frames: {vid_result['ai_flagged_frames']}")
            print(f"[PASS] Average Frame Score: {vid_result['average_frame_score']:.4f}")
            print(f"[PASS] Median Frame Score: {vid_result['median_frame_score']:.4f}")
            print(f"[PASS] Frame Timestamps: {vid_result['frame_timestamps_seconds']}")
            print(f"[PASS] Audio Track Present: {vid_result['audio_track_present']}, Status: {vid_result['audio_status']}")
            if vid_result["audio_analysis"]:
                print(f"  Audio Verdict: {vid_result['audio_analysis']['verdict']}, Score: {vid_result['audio_analysis']['score']:.4f}")

            # Verify timestamps calculation and 8-frame cap
            duration = vid_result["duration_seconds"]
            timestamps = vid_result["frame_timestamps_seconds"]
            assert len(timestamps) <= 8, f"Frames exceeded 8-frame cap: {len(timestamps)}"
            assert len(timestamps) == vid_result["frames_analyzed"]
            # Check frame 1 has valid image detector outputs
            f0 = vid_result["frame_results"][0]
            assert "verdict" in f0 and f0["verdict"] in ("AI-generated", "Likely real")
            assert "score" in f0 and 0.0 <= f0["score"] <= 1.0
            assert "logit" in f0
            assert "explanation" in f0 and "crop_png_base64" in f0["explanation"]
            print("[PASS] Frame-level image analysis and explanations verified on extracted video frames.")
            audit_results["video"]["sample_video"] = True

        # Test video without audio track
        print("\n--- Testing Video Without Audio Track ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            no_audio_video = Path(temp_dir) / "no_audio.mp4"
            # Generate 4-second video without audio using lavfi testsrc
            cmd = [
                shutil.which("ffmpeg"), "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
                "-pix_fmt", "yuv420p",
                str(no_audio_video),
            ]
            subprocess.run(cmd, check=True)
            res_no_audio = video_detector.predict(no_audio_video)
            print(f"[PASS] Video without audio processed: frames={res_no_audio['frames_analyzed']}, audio_status={res_no_audio['audio_status']}")
            assert res_no_audio["audio_track_present"] is False
            assert res_no_audio["audio_status"] == "no_audio_stream"
            assert res_no_audio["audio_analysis"] is None
            audit_results["video"]["no_audio"] = True

        # Test video formats: MP4, MKV, AVI, MOV
        print("\n--- Testing Video Formats (MP4, MKV, AVI, MOV) ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for ext in ["mp4", "mkv", "avi", "mov"]:
                test_vid = temp_path / f"test_fmt.{ext}"
                cmd = [
                    shutil.which("ffmpeg"), "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264" if ext != "avi" else "mpeg4",
                    "-c:a", "aac" if ext in ("mp4", "mov", "mkv") else "pcm_s16le",
                    str(test_vid),
                ]
                subprocess.run(cmd, check=True)
                v_res = video_detector.predict(test_vid)
                print(f"[PASS] Video Format .{ext.upper()}: frames={v_res['frames_analyzed']}, duration={v_res['duration_seconds']}s, audio_status={v_res['audio_status']}")
                assert v_res["frames_analyzed"] >= 1
                assert v_res["audio_track_present"] is True

        # Test video with duration > 16s (max 8 frames cap check)
        print("\n--- Testing 20-Second Video (Cap at 8 Frames) ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            long_vid = Path(temp_dir) / "long_20s.mp4"
            cmd = [
                shutil.which("ffmpeg"), "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=duration=20:size=320x240:rate=5",
                "-pix_fmt", "yuv420p",
                str(long_vid),
            ]
            subprocess.run(cmd, check=True)
            long_res = video_detector.predict(long_vid)
            print(f"[PASS] 20s Video: frames={long_res['frames_analyzed']} (cap=8), timestamps={long_res['frame_timestamps_seconds']}")
            assert long_res["frames_analyzed"] == 8
            assert len(long_res["frame_timestamps_seconds"]) == 8

        # Test Corrupted Video File
        print("\n--- Testing Corrupted Video File Handling ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt_vid = Path(temp_dir) / "corrupt.mp4"
            corrupt_vid.write_bytes(b"CORRUPTED_VIDEO_HEADER_DATA")
            try:
                video_detector.predict(corrupt_vid)
                print("[FAIL] Corrupted video was NOT rejected!")
                audit_results["video"]["corrupt_rejected"] = False
            except ValueError as exc:
                print(f"[PASS] Corrupted video correctly rejected: '{exc}'")
                audit_results["video"]["corrupt_rejected"] = True

    print("\n" + "=" * 70)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
