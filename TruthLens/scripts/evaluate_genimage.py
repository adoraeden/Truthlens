r"""Evaluate TruthLens's production image detector on the local GenImage sample.

This is a fixed-threshold evaluation only. It does not train a model or calibrator.
Run from the repository root with: .\.venv\Scripts\python.exe scripts\evaluate_genimage.py
"""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.image_detector import ImageDetector, LOGIT_THRESHOLD


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "genimage_validation_sample"
DEFAULT_REPORTS = ROOT / "reports"
GENERATORS = (
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
)
LABEL_MAP = {"ai": "ai", "nature": "nature"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PREVIOUS_EVALUATION_DIRS = ("test_images", "new_real", "new_fake", "mistake_review")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(dataset_root, expected_per_class=25, generators=GENERATORS):
    """Validate manifest rows, file hashes, directory layout, and per-class counts."""
    dataset_root = Path(dataset_root).resolve()
    manifest_path = dataset_root / "manifest.csv"
    required = {"archive_path", "generator", "label", "sha256"}
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{manifest_path} must contain columns {sorted(required)}")
        rows = list(reader)

    expected_counts = {(generator, label): 0 for generator in generators for label in LABEL_MAP}
    seen_paths, seen_hashes = set(), set()
    validated = []
    for line, row in enumerate(rows, start=2):
        relative = Path(row["archive_path"])
        generator = row["generator"].strip()
        label = row["label"].strip().lower()
        expected_hash = row["sha256"].strip().lower()
        if generator not in generators or label not in LABEL_MAP:
            raise ValueError(f"Unknown generator/label on {manifest_path.name}:{line}: {generator!r}/{label!r}")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe archive_path on {manifest_path.name}:{line}")
        if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
            raise ValueError(f"Invalid SHA-256 on {manifest_path.name}:{line}")
        if relative.as_posix() in seen_paths:
            raise ValueError(f"Duplicate archive_path in manifest: {relative.as_posix()}")
        if expected_hash in seen_hashes:
            raise ValueError(f"Duplicate sample hash in manifest: {expected_hash}")
        if len(relative.parts) < 3 or relative.parts[0] != generator or relative.parts[1].lower() != label:
            raise ValueError(f"Manifest path does not match generator/label on line {line}: {relative}")
        path = (dataset_root / relative).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes dataset directory: {relative}") from exc
        if not path.is_file():
            raise ValueError(f"Manifest image is missing: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Manifest SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual_hash}")
        seen_paths.add(relative.as_posix())
        seen_hashes.add(expected_hash)
        expected_counts[(generator, label)] += 1
        validated.append({
            "path": path,
            "filename": relative.as_posix(),
            "generator": generator,
            "ground_truth": LABEL_MAP[label],
            "sha256": actual_hash,
        })

    missing_or_wrong = {
        f"{generator}/{label}": count
        for (generator, label), count in expected_counts.items()
        if count != expected_per_class
    }
    if missing_or_wrong:
        raise ValueError(f"Expected {expected_per_class} images per generator/class; found {missing_or_wrong}")
    expected_total = len(generators) * len(LABEL_MAP) * expected_per_class
    if len(validated) != expected_total:
        raise ValueError(f"Expected {expected_total} manifest rows, found {len(validated)}")

    disk_paths = {
        path.resolve()
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    manifest_paths = {sample["path"] for sample in validated}
    if disk_paths != manifest_paths:
        unlisted = sorted(str(path.relative_to(dataset_root)) for path in disk_paths - manifest_paths)
        absent = sorted(str(path.relative_to(dataset_root)) for path in manifest_paths - disk_paths)
        raise ValueError(f"Manifest/directory image mismatch; unlisted={unlisted[:10]}, absent={absent[:10]}")
    return validated


def calculate_metrics(records):
    """Calculate AI-positive metrics on successfully scored rows and expose coverage."""
    total_counts = Counter(row["ground_truth"] for row in records)
    scored = [row for row in records if row.get("predicted_label") in LABEL_MAP.values()]
    scored_counts = Counter(row["ground_truth"] for row in scored)
    errors = len(records) - len(scored)
    tp = fp = tn = fn = 0
    for row in scored:
        positive_truth = row["ground_truth"] == "ai"
        positive_prediction = row["predicted_label"] == "ai"
        if positive_truth and positive_prediction:
            tp += 1
        elif positive_truth:
            fn += 1
        elif positive_prediction:
            fp += 1
        else:
            tn += 1

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "sample_counts": {
            "total": len(records),
            "scored": len(scored),
            "inference_errors": errors,
            "coverage": ratio(len(scored), len(records)),
            "classes_total": {label: total_counts[label] for label in ("ai", "nature")},
            "classes_scored": {label: scored_counts[label] for label in ("ai", "nature")},
            "classes_with_inference_errors": {
                label: total_counts[label] - scored_counts[label] for label in ("ai", "nature")
            },
        },
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "accuracy": ratio(tp + tn, len(scored)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "false_positive_rate": ratio(fp, fp + tn),
            "false_negative_rate": ratio(fn, fn + tp),
        },
    }


def index_previous_evaluation_hashes(root=ROOT):
    """Index local prior image-evaluation directories by exact content hash."""
    index = defaultdict(list)
    files_seen = 0
    for directory in PREVIOUS_EVALUATION_DIRS:
        base = Path(root) / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                files_seen += 1
                try:
                    index[sha256_file(path)].append(str(path.resolve()))
                except OSError:
                    continue
    return index, files_seen


def find_hash_overlap(samples, previous_hashes):
    return [
        {"filename": sample["filename"], "sha256": sample["sha256"], "previous_paths": previous_hashes[sample["sha256"]]}
        for sample in samples
        if sample["sha256"] in previous_hashes
    ]


def evaluate_samples(detector, samples):
    records = []
    for sample in samples:
        started = time.perf_counter()
        record = {
            "filename": sample["filename"],
            "generator": sample["generator"],
            "ground_truth": sample["ground_truth"],
            "predicted_label": "",
            "raw_model_logit": "",
            "decision_threshold": "",
            "threshold_relative_score_uncalibrated": "",
            "score_method": "",
            "score_is_calibrated_probability": False,
            "prediction_latency_ms": "",
            "sha256": sample["sha256"],
            "inference_error": "",
        }
        try:
            result = detector.predict(sample["path"])
            logit = float(result["logit"])
            threshold = float(result["decision_threshold"])
            if not math.isfinite(logit) or not math.isfinite(threshold):
                raise ValueError("Detector returned a non-finite logit or threshold")
            verdict_map = {"AI-generated": "ai", "Likely real": "nature"}
            predicted = verdict_map.get(result.get("verdict"))
            expected = "ai" if logit >= threshold else "nature"
            if predicted is None or predicted != expected:
                raise ValueError("Detector verdict does not match its raw-logit threshold rule")
            record.update({
                "predicted_label": predicted,
                "raw_model_logit": f"{logit:.10g}",
                "decision_threshold": f"{threshold:.10g}",
                "threshold_relative_score_uncalibrated": result.get("score", ""),
                "score_method": result.get("score_method", ""),
                "score_is_calibrated_probability": result.get("score_calibrated_probability") is True,
            })
        except Exception as exc:  # Keep a row for every sample, including model/decode failures.
            record["inference_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["prediction_latency_ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
        records.append(record)
    return records


def aggregate(records, generators=GENERATORS):
    overall = calculate_metrics(records)
    by_generator = {
        generator: calculate_metrics([row for row in records if row["generator"] == generator])
        for generator in generators
    }
    return {"overall": overall, "by_generator": by_generator}


def write_predictions(path, records):
    fields = [
        "filename", "generator", "ground_truth", "predicted_label", "raw_model_logit",
        "decision_threshold", "threshold_relative_score_uncalibrated", "score_method",
        "score_is_calibrated_probability", "prediction_latency_ms", "sha256", "inference_error",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _markdown_metrics(name, report):
    counts, confusion, metrics = report["sample_counts"], report["confusion"], report["metrics"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    return [
        f"### {name}",
        f"Samples: {counts['total']} total, {counts['scored']} scored, {counts['inference_errors']} errors; coverage {fmt(counts['coverage'])}",
        f"Class counts (total): AI={counts['classes_total']['ai']}, nature={counts['classes_total']['nature']}; "
        f"scored: AI={counts['classes_scored']['ai']}, nature={counts['classes_scored']['nature']}; "
        f"errors: AI={counts['classes_with_inference_errors']['ai']}, nature={counts['classes_with_inference_errors']['nature']}",
        f"Confusion (AI positive): TP={confusion['tp']}, FP={confusion['fp']}, TN={confusion['tn']}, FN={confusion['fn']}",
        f"Accuracy={fmt(metrics['accuracy'])}; precision={fmt(metrics['precision'])}; recall={fmt(metrics['recall'])}; "
        f"specificity={fmt(metrics['specificity'])}; F1={fmt(metrics['f1'])}; balanced accuracy={fmt(metrics['balanced_accuracy'])}; "
        f"FPR={fmt(metrics['false_positive_rate'])}; FNR={fmt(metrics['false_negative_rate'])}",
        "",
    ]


def write_report(report_path, dataset_root, result, overlap, previous_files, failures, threshold):
    try:
        dataset_display = Path(dataset_root).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        dataset_display = str(Path(dataset_root).resolve())
    lines = [
        "# TruthLens GenImage Validation Report", "",
        "## Evaluation setup", "",
        f"- Dataset: `{dataset_display}` (manifest labels; AI is positive, nature is negative).",
        f"- Production decision threshold: raw logit ≥ {threshold:g}.",
        "- Detector: existing Community Forensics ViT-S/16 via `ImageDetector.predict`; no model or preprocessing changes.",
        "- `threshold_relative_score_uncalibrated` is sigmoid(logit − threshold), not a calibrated probability.",
        "- Calibrated probabilities are not used for predictions or reported as evaluation scores.",
        "", "## Overall metrics", "",
    ]
    lines.extend(_markdown_metrics("Overall", result["overall"]))
    lines.extend(["## Per-generator metrics", ""])
    for generator, metrics_report in result["by_generator"].items():
        lines.extend(_markdown_metrics(generator, metrics_report))
    lines.extend([
        "## Exact-hash overlap with local prior evaluation images", "",
        f"Indexed {previous_files} files under `{', '.join(PREVIOUS_EVALUATION_DIRS)}`. "
        f"Exact GenImage hash overlaps: **{len(overlap)}**.",
    ])
    if overlap:
        for item in overlap:
            lines.append(f"- `{item['filename']}` ({item['sha256']}): " + ", ".join(f"`{path}`" for path in item["previous_paths"]))
    else:
        lines.append("- No byte-identical sample hashes were found in those local evaluation-image directories.")
    lines.extend(["", "## Processing validation", "", f"Successfully scored {result['overall']['sample_counts']['scored']} of {result['overall']['sample_counts']['total']} manifest samples."])
    if failures:
        lines.extend(["", "Inference failures (rows retained in the predictions CSV):"])
        lines.extend(f"- `{item['filename']}`: {item['inference_error']}" for item in failures)
    else:
        lines.append("All manifest samples were processed successfully.")
    lines.append("")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(dataset_root=DEFAULT_DATASET, reports_dir=DEFAULT_REPORTS, detector=None):
    samples = validate_manifest(dataset_root)
    # The production detector (including model/checkpoint initialization) is constructed once.
    detector = detector if detector is not None else ImageDetector()
    records = evaluate_samples(detector, samples)
    metrics_report = aggregate(records)
    previous_hashes, previous_files = index_previous_evaluation_hashes()
    overlap = find_hash_overlap(samples, previous_hashes)
    failures = [row for row in records if row["inference_error"]]
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "genimage_validation_predictions.csv"
    metrics_path = output_dir / "genimage_validation_metrics.json"
    markdown_path = output_dir / "genimage_validation_report.md"
    write_predictions(predictions_path, records)
    full_metrics = {
        "dataset": str(Path(dataset_root)),
        "positive_class": "ai",
        "negative_class": "nature",
        "threshold": LOGIT_THRESHOLD,
        "score_definition": "sigmoid(raw_logit - decision_threshold); uncalibrated threshold-relative score, not probability",
        "model": "Community Forensics ViT-S/16",
        "prediction_count": len(records),
        "successful_prediction_count": len(records) - len(failures),
        "inference_error_count": len(failures),
        **metrics_report,
        "exact_hash_overlap_count": len(overlap),
        "exact_hash_overlaps": overlap,
        "previous_evaluation_files_indexed": previous_files,
        "inference_errors": [{"filename": row["filename"], "error": row["inference_error"]} for row in failures],
    }
    metrics_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    write_report(markdown_path, dataset_root, metrics_report, overlap, previous_files, failures, LOGIT_THRESHOLD)
    return {"predictions": predictions_path, "metrics": metrics_path, "report": markdown_path, **full_metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    result = run_evaluation(args.dataset, args.reports_dir)
    print(f"Processed {result['successful_prediction_count']}/{result['prediction_count']} images successfully.")
    print(f"Exact hash overlap with local prior evaluation images: {result['exact_hash_overlap_count']}.")
    print(f"Predictions: {result['predictions']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Report: {result['report']}")
    if result["inference_error_count"]:
        print("One or more images failed; see the predictions CSV and report. Evaluation exits with status 2.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
