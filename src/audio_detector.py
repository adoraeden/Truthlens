import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PELLA_DIR = PROJECT_ROOT / "models" / "pellav2"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
ANALYSIS_WINDOW_SECONDS = 4.0
MIN_AUDIO_SECONDS = 0.25

if str(PELLA_DIR) not in sys.path:
    sys.path.insert(0, str(PELLA_DIR))

import pellav2_infer as pella


class AudioDetector:
    def __init__(self):
        torch.set_num_threads(8)
        self.device = torch.device("cpu")

        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise RuntimeError("FFmpeg and FFprobe must be installed and available in PATH.")

        self.model = pella.Detector().to(self.device)

        checkpoint = torch.load(
            PELLA_DIR / "pellav2_detector.pt",
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _get_duration(self, audio_path):
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            duration = 0.0

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("The audio file is invalid or has no readable audio stream.")
        if duration < MIN_AUDIO_SECONDS:
            raise ValueError("Audio is too short to analyze; provide at least 0.25 seconds.")
        return duration

    def predict(
        self,
        audio_path,
        *,
        source_duration_seconds=None,
        analysis_start_seconds=None,
    ):
        audio_path = Path(audio_path)

        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError("Unsupported audio format. Use WAV, MP3, M4A, FLAC, OGG, or AAC.")
        if not audio_path.is_file():
            raise ValueError("The audio file could not be found.")

        probed_duration = self._get_duration(audio_path)
        source_duration = float(
            probed_duration if source_duration_seconds is None else source_duration_seconds
        )
        if not math.isfinite(source_duration) or source_duration <= 0:
            raise ValueError("The audio duration is invalid.")

        crop_start = (
            max(0.0, (source_duration - ANALYSIS_WINDOW_SECONDS) / 2)
            if analysis_start_seconds is None
            else max(0.0, float(analysis_start_seconds))
        )
        is_center_crop = source_duration > ANALYSIS_WINDOW_SECONDS

        # Seek and cap decoding in FFmpeg so long recordings do not need to be
        # fully converted and loaded into memory just to analyze one window.
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "analysis_window.wav"
            command = [self.ffmpeg, "-y", "-v", "error"]
            if is_center_crop:
                command.extend(["-ss", f"{crop_start:.6f}"])
            command.extend([
                "-i", str(audio_path),
                "-t", str(ANALYSIS_WINDOW_SECONDS),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ])

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ValueError("Audio conversion failed. Check that the file is valid and supported.") from exc

            if result.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
                raise ValueError("Audio conversion failed. Check that the file is valid and supported.")

            try:
                waveform = pella.load_clip(str(wav_path))
            except Exception as exc:
                raise ValueError("The audio stream could not be decoded for analysis.") from exc

            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("The audio stream is empty or invalid.")
            if float(np.max(np.abs(waveform))) < 1e-8:
                raise ValueError("The recording is silent and cannot be analyzed.")

            audio_tensor = torch.from_numpy(waveform)[None].to(self.device)
            with torch.inference_mode():
                logit = self.model(audio_tensor)
                score = torch.sigmoid(logit).item()

        analyzed_seconds = min(source_duration, ANALYSIS_WINDOW_SECONDS)
        if is_center_crop:
            note = (
                "Only the 4-second center crop was analyzed; other parts of the "
                "recording may contain different content. Re-recorded synthetic speech may be missed."
            )
            region = "center_crop"
        else:
            note = (
                "The full recording was analyzed; recordings shorter than 4 seconds "
                "are zero-padded to the model's 4-second input. Re-recorded synthetic speech may be missed."
            )
            region = "full_recording"

        return {
            "verdict": "AI-generated" if score >= 0.5 else "Likely real",
            "score": score,
            "duration_analyzed_seconds": round(analyzed_seconds, 2),
            "source_duration_seconds": round(source_duration, 2),
            "model_input_seconds": int(ANALYSIS_WINDOW_SECONDS),
            "analysis_region": region,
            "analysis_start_seconds": round(crop_start, 2),
            "note": note,
            "model": "PellaV2 (Wav2Vec2-BERT 2.0)",
        }
