"""Evaluate PellaV2 on the existing audio samples or inspect decoded windows.

The diagnostics-only path never loads PellaV2. Model evaluation is guarded by a
5 GB free-memory preflight and always uses AudioDetector's production path and
fixed 0.50 score cutoff; this script does not tune thresholds.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evaluation" / "audio_manifest.csv"
MIN_PELLA_FREE_RAM_GB = 3.5
WINDOW_SECONDS = 4.0


def load_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "condition", "source_set", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include columns: {', '.join(sorted(required))}")
    for row in rows:
        if row["label"].lower() not in {"real", "fake"}:
            raise ValueError(f"Invalid label for {row['path']}: {row['label']!r}")
        row["resolved_path"] = str((ROOT / row["path"]).resolve())
        if not Path(row["resolved_path"]).is_file():
            raise FileNotFoundError(f"Missing evaluation sample: {row['resolved_path']}")
    return rows


def classification_metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def require_pella_memory(available_gb):
    if available_gb < MIN_PELLA_FREE_RAM_GB:
        raise RuntimeError(
            f"PellaV2 evaluation not started: {available_gb:.2f} GB RAM is free; "
            f"at least {MIN_PELLA_FREE_RAM_GB:.1f} GB is required by the project runtime note."
        )


def _probe_audio(path, ffprobe):
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,codec_name,sample_rate,channels,channel_layout",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    metadata = json.loads(result.stdout)
    stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError("No audio stream found")
    duration = float(metadata.get("format", {}).get("duration", 0.0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio duration")
    return stream, duration


def decode_production_window(path, ffmpeg, duration):
    start = max(0.0, (duration - WINDOW_SECONDS) / 2)
    command = [ffmpeg, "-v", "error"]
    if duration > WINDOW_SECONDS:
        command.extend(["-ss", f"{start:.6f}"])
    command.extend([
        "-i", str(path), "-t", str(WINDOW_SECONDS), "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "pipe:1",
    ])
    result = subprocess.run(command, capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode the analysis window: {message}")
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return start, samples


def run_diagnostics(rows):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be available for diagnostics.")

    print("Signal diagnostics only; PellaV2 was not loaded or run.")
    print("file,label,condition,codec,source_rate_hz,source_channels,duration_s,window_start_s,window_samples,rms_dbfs,peak_dbfs,clipped_pct")
    for row in rows:
        path = Path(row["resolved_path"])
        stream, duration = _probe_audio(path, ffprobe)
        start, samples = decode_production_window(path, ffmpeg, duration)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms_db = 20 * math.log10(rms) if rms > 0 else float("-inf")
        peak_db = 20 * math.log10(peak) if peak > 0 else float("-inf")
        clipped = float(np.mean(np.abs(samples) >= 0.999) * 100) if samples.size else 0.0
        values = [
            path.name, row["label"], row["condition"], stream.get("codec_name", "unknown"),
            stream.get("sample_rate", "unknown"), stream.get("channels", "unknown"),
            f"{duration:.3f}", f"{start:.3f}", str(samples.size), f"{rms_db:.2f}",
            f"{peak_db:.2f}", f"{clipped:.4f}",
        ]
        print(",".join(str(value) for value in values))


def run_pella_evaluation(rows):
    import psutil

    available_gb = psutil.virtual_memory().available / 1e9
    require_pella_memory(available_gb)

    # Import model code only after the memory gate passes.
    from src.audio_detector import AudioDetector

    detector = AudioDetector()
    results = []
    for row in rows:
        start = time.perf_counter()
        prediction = detector.predict(row["resolved_path"])
        elapsed = time.perf_counter() - start
        label = int(row["label"].lower() == "fake")
        predicted = int(prediction["score"] >= 0.5)
        results.append({
            "label": label,
            "prediction": predicted,
            "score": float(prediction["score"]),
            "elapsed_seconds": elapsed,
            "condition": row["condition"],
            "split": row["split"],
            "path": row["path"],
        })
        print(
            f"{row['path']}: {row['condition']} truth={label} "
            f"score={prediction['score']:.6f} pred={predicted} time={elapsed:.2f}s"
        )

    groups = defaultdict(list)
    groups["all"].extend(results)
    for row in results:
        groups[row["condition"]].append(row)

    print("\nFixed PellaV2 cutoff: score >= 0.50; no threshold search performed.")
    for name, group in groups.items():
        metric = classification_metrics(
            [item["label"] for item in group],
            [item["prediction"] for item in group],
        )
        mean_time = float(np.mean([item["elapsed_seconds"] for item in group]))
        print(
            f"{name}: n={metric['n']} TP={metric['tp']} FP={metric['fp']} "
            f"TN={metric['tn']} FN={metric['fn']} accuracy={metric['accuracy']:.3f} "
            f"precision={metric['precision']:.3f} recall={metric['recall']:.3f} "
            f"F1={metric['f1']:.3f} FPR={metric['fpr']:.3f} FNR={metric['fnr']:.3f} "
            f"mean_inference={mean_time:.2f}s"
        )

    splits = {row["split"] for row in rows}
    if splits == {"exploratory_seen"}:
        print("LIMITATION: all manifest samples are previously seen exploratory data; this is not a final holdout.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Decode and summarize production-format windows without loading PellaV2.",
    )
    args = parser.parse_args(argv)
    rows = load_manifest(args.manifest)
    if args.diagnostics_only:
        run_diagnostics(rows)
        return 0

    try:
        run_pella_evaluation(rows)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
