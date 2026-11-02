import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class VideoDetector:
    def __init__(self, image_detector, audio_detector, max_frames=8):
        self.image_detector = image_detector
        self.audio_detector = audio_detector
        self.max_frames = max(1, int(max_frames))

        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise RuntimeError("FFmpeg and FFprobe must be installed and available in PATH.")

    def _probe_media(self, video_path):
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v", "error",
                    "-show_entries", "format=duration:stream=codec_type,duration",
                    "-of", "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            metadata = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ValueError("The video file is invalid or cannot be read.") from exc

        streams = metadata.get("streams", [])
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        if not video_streams:
            raise ValueError("The uploaded file does not contain a video stream.")

        duration = None
        candidates = [
            video_streams[0].get("duration"),
            metadata.get("format", {}).get("duration"),
        ]
        for candidate in candidates:
            try:
                candidate_duration = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate_duration) and candidate_duration > 0:
                duration = candidate_duration
                break
        if duration is None:
            raise ValueError("The video duration could not be determined.")

        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        return duration, has_audio

    def _extract_frame(self, video_path, timestamp, output_path):
        try:
            result = subprocess.run(
                [
                    self.ffmpeg,
                    "-y", "-v", "error",
                    "-ss", f"{timestamp:.6f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0

    def _extract_audio(self, video_path, output_path, duration):
        audio_start = max(0.0, (duration - 4.0) / 2)
        command = [self.ffmpeg, "-y", "-v", "error"]
        if duration > 4.0:
            command.extend(["-ss", f"{audio_start:.6f}"])
        command.extend([
            "-i", str(video_path),
            "-t", "4",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(output_path),
        ])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, audio_start
        usable = result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0
        return usable, audio_start

    def predict(self, video_path):
        video_path = Path(video_path)
        if not video_path.is_file():
            raise ValueError("The video file could not be found.")

        duration, has_audio = self._probe_media(video_path)

        # Midpoints of evenly sized time intervals avoid relying on the first
        # frame and distribute the limited samples across the full runtime.
        frame_count = min(self.max_frames, max(1, math.ceil(duration / 5.0)))
        timestamps = [
            duration * (index + 0.5) / frame_count
            for index in range(frame_count)
        ]
        frame_results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            for index, timestamp in enumerate(timestamps):
                frame_path = temp_dir / f"frame_{index:03d}.jpg"
                if not self._extract_frame(video_path, timestamp, frame_path):
                    continue

                try:
                    with Image.open(frame_path) as image:
                        image.load()
                        result = self.image_detector.predict(image.copy())
                except (UnidentifiedImageError, OSError) as exc:
                    raise ValueError("A sampled video frame could not be decoded.") from exc

                result["timestamp_seconds"] = round(float(timestamp), 2)
                frame_results.append(result)

            if not frame_results:
                raise ValueError("No usable video frames could be extracted.")

            audio_result = None
            if not has_audio:
                audio_status = "no_audio_stream"
            else:
                audio_path = temp_dir / "video_audio.wav"
                audio_ok, audio_start = self._extract_audio(video_path, audio_path, duration)
                if audio_ok:
                    try:
                        audio_result = self.audio_detector.predict(
                            audio_path,
                            source_duration_seconds=duration,
                            analysis_start_seconds=audio_start,
                        )
                        audio_status = "analyzed"
                    except ValueError:
                        # A broken audio stream should not invalidate usable
                        # visual evidence from the same video.
                        audio_status = "audio_stream_unreadable"
                else:
                    audio_status = "audio_stream_unreadable"

        frame_scores = [item["score"] for item in frame_results]
        ai_frames = sum(item["verdict"] == "AI-generated" for item in frame_results)

        return {
            "duration_seconds": round(duration, 2),
            "frames_requested": len(timestamps),
            "frames_analyzed": len(frame_results),
            "frame_timestamps_seconds": [item["timestamp_seconds"] for item in frame_results],
            "frame_sampling_method": "Midpoint of uniform temporal intervals; sampled frames do not cover every moment.",
            "ai_flagged_frames": ai_frames,
            "average_frame_score": float(np.mean(frame_scores)),
            "frame_score_note": "Mean of sampled-frame model scores; not a whole-video probability.",
            "frame_results": frame_results,
            "audio_track_present": has_audio,
            "audio_status": audio_status,
            "audio_analysis": audio_result,
            "interpretation": (
                "Video frames and audio are evaluated separately. Only the listed "
                "frame timestamps and, when available, a four-second audio window "
                "were analyzed. These samples do not establish the authenticity "
                "of the complete video."
            ),
        }
