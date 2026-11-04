import math
import shutil
import subprocess
import sys
import tempfile
import ctypes
import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PELLA_DIR = PROJECT_ROOT / "models" / "pellav2"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
ANALYSIS_WINDOW_SECONDS = 4.0
MIN_AUDIO_SECONDS = 0.25
PELLA_MIN_AVAILABLE_GB = float(os.getenv("TRUTHLENS_PELLA_MIN_RAM_GB", "3.5"))


def available_memory_gb():
    """Return available physical RAM, or zero when it cannot be determined."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 1e9
        return 0.0

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return available_pages * page_size / 1e9
    except (AttributeError, OSError, ValueError):
        return 0.0


def pella_can_load(available_gb):
    try:
        available_gb = float(available_gb)
    except (TypeError, ValueError):
        return False
    return math.isfinite(available_gb) and available_gb >= PELLA_MIN_AVAILABLE_GB

if str(PELLA_DIR) not in sys.path:
    sys.path.insert(0, str(PELLA_DIR))

# pellav2_infer is imported lazily in AudioDetector.__init__ so a missing
# PellaV2 folder does not crash app startup (e.g. on Render).


class AudioDetector:
    def __init__(self):
        torch.set_num_threads(8)
        self.device = torch.device("cpu")

        available_gb = available_memory_gb()
        if not pella_can_load(available_gb):
            raise RuntimeError(
                f"PellaV2 requires at least {PELLA_MIN_AVAILABLE_GB:.1f} GB of free RAM to load, "
                f"but only {available_gb:.2f} GB is currently available. "
                f"Close other applications to free memory, then restart the backend. "
                f"You can lower this threshold by setting the environment variable "
                f"TRUTHLENS_PELLA_MIN_RAM_GB (current default: {PELLA_MIN_AVAILABLE_GB:.1f})."
            )

        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise RuntimeError("FFmpeg and FFprobe must be installed and available in PATH.")

        checkpoint_path = PELLA_DIR / "pellav2_detector.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(
                f"PellaV2 checkpoint not found at {checkpoint_path}. "
                "Audio analysis is unavailable."
            )
        try:
            import pellav2_infer
        except ImportError as exc:
            raise RuntimeError(
                f"PellaV2 module could not be imported ({exc}). Audio analysis is unavailable."
            ) from exc
        self._pella = pellav2_infer

        self.model = self._pella.Detector().to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
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
        already_cropped=False,
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
        # VideoDetector can pass an already-extracted window while retaining
        # the original video's duration and crop offset for result metadata.
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "analysis_window.wav"
            command = [self.ffmpeg, "-y", "-v", "error"]
            if is_center_crop and not already_cropped:
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
                waveform = self._pella.load_clip(str(wav_path))
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