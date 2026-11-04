"""Compare frame-level video summary methods on labeled local videos.

Uses the production VideoDetector sampling/extraction helpers and the existing
ImageDetector preprocessing/inference. A free-RAM preflight runs before model
import or construction. Method metrics are exploratory, not threshold tuning.
"""

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.video_detector import VideoDetector

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evaluation" / "video_manifest.csv"
MIN_IMAGE_FREE_RAM_GB = 1.5
FRAME_SCORE_CUTOFF = 0.5


def load_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "source_set", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include columns: {', '.join(sorted(required))}")
    for row in rows:
        if row["label"].lower() not in {"real", "fake"}:
            raise ValueError(f"Invalid label for {row['path']}: {row['label']!r}")
        row["resolved_path"] = str((ROOT / row["path"]).resolve())
        if not Path(row["resolved_path"]).is_file():
            raise FileNotFoundError(f"Missing video: {row['resolved_path']}")
    return rows


def summarize_methods(scores):
    if not scores:
        raise ValueError("At least one frame score is required")
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "flagged_fraction": float(np.mean(np.asarray(scores) >= FRAME_SCORE_CUTOFF)),
    }


def classification_metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
    }


def require_image_memory(available_gb):
    if available_gb < MIN_IMAGE_FREE_RAM_GB:
        raise RuntimeError(
            f"Video image evaluation not started: {available_gb:.2f} GB RAM is free; "
            f"at least {MIN_IMAGE_FREE_RAM_GB:.1f} GB is required by this evaluation guard."
        )


def run(rows):
    import psutil

    available_gb = psutil.virtual_memory().available / 1e9
    require_image_memory(available_gb)
    from src.image_detector import ImageDetector
    from PIL import Image

    image_detector = ImageDetector()
    video_detector = object.__new__(VideoDetector)
    video_detector.image_detector = image_detector
    video_detector.max_frames = 8
    video_detector.ffmpeg = shutil.which("ffmpeg")
    video_detector.ffprobe = shutil.which("ffprobe")
    if not video_detector.ffmpeg or not video_detector.ffprobe:
        raise RuntimeError("FFmpeg and FFprobe are required")

    per_video = []
    for row in rows:
        path = Path(row["resolved_path"])
        duration, _ = video_detector._probe_media(path)
        timestamps = video_detector._sample_timestamps(duration)
        scores, times = [], []
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            for index, timestamp in enumerate(timestamps):
                frame = Path(folder) / f"frame_{index:03d}.jpg"
                if not video_detector._extract_frame(path, timestamp, frame):
                    continue
                start = time.perf_counter()
                with Image.open(frame) as image:
                    result = image_detector.predict(image.copy())
                times.append(time.perf_counter() - start)
                scores.append(float(result["score"]))
        summaries = summarize_methods(scores)
        item = {
            "path": row["path"], "label": int(row["label"].lower() == "fake"),
            "split": row["split"], "duration_seconds": duration,
            "frames_analyzed": len(scores), "frame_timestamps_seconds": timestamps[:len(scores)],
            "frame_scores": scores, "mean_frame_inference_seconds": float(np.mean(times)) if times else 0.0,
            "total_frame_inference_seconds": float(np.sum(times)), **summaries,
        }
        per_video.append(item)
        print(f"{item['path']}: truth={item['label']} frames={len(scores)} "
              f"scores={[round(s, 4) for s in scores]} "
              f"mean={item['mean']:.4f} median={item['median']:.4f} "
              f"flagged_fraction={item['flagged_fraction']:.3f} "
              f"mean_frame_time={item['mean_frame_inference_seconds']:.3f}s")

    print(f"\nFrame cutoff fixed at {FRAME_SCORE_CUTOFF:.2f}; aggregation methods compared only.")
    for method in ("mean", "median", "flagged_fraction"):
        metrics = classification_metrics(
            [item["label"] for item in per_video],
            [int(item[method] >= FRAME_SCORE_CUTOFF) for item in per_video],
        )
        print(f"{method}: {metrics}")
    if {row["split"] for row in rows} == {"exploratory_seen"}:
        print("LIMITATION: samples are previously seen smoke-test videos, not a final holdout.")
        print("Audio is deliberately not included in these visual-only measurements.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        run(load_manifest(args.manifest))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
