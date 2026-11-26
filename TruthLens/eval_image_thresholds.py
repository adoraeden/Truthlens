"""Recompute fixed-cutoff image metrics from saved labeled logits (no model load).

AI-generated is the positive class. The CSVs are exploratory development data,
not an untouched final holdout. This utility deliberately does not choose or
write a production threshold.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = (
    ROOT / "eval_results_web.csv",
    ROOT / "independent_image_eval.csv",
)
DEFAULT_THRESHOLDS = (1.359375, 2.0)


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"logit"}.issubset(rows[0]):
        raise ValueError(f"{path} has no labeled logit rows")
    label_column = "truth" if "truth" in rows[0] else "actual" if "actual" in rows[0] else None
    if label_column is None:
        raise ValueError(f"{path} must include truth or actual labels")

    parsed = []
    for row in rows:
        label = str(row[label_column]).strip().lower()
        if label in {"1", "fake", "ai", "ai-generated", "synthetic"}:
            positive = 1
        elif label in {"0", "real", "genuine"}:
            positive = 0
        else:
            raise ValueError(f"Unknown label {row[label_column]!r} in {path}")
        logit = float(row["logit"])
        if not math.isfinite(logit):
            raise ValueError(f"Non-finite logit for {row.get('file', 'sample')} in {path}")
        parsed.append((positive, logit))
    return parsed


def metrics(rows, threshold):
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    tp = fp = tn = fn = 0
    for truth, logit in rows:
        predicted_fake = logit >= threshold
        if truth and predicted_fake:
            tp += 1
        elif truth:
            fn += 1
        elif predicted_fake:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = tp + fp + tn + fn
    return {
        "n": n, "real_n": tn + fp, "ai_n": tp + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def exact_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_exact_overlap(thresholds):
    baseline_dir = ROOT / "test_images"
    followup_dirs = (ROOT / "new_real", ROOT / "new_fake")
    if not baseline_dir.is_dir() or not all(folder.is_dir() for folder in followup_dirs):
        return
    baseline = {exact_hash(path): path for path in baseline_dir.iterdir() if path.is_file()}
    followup = {
        exact_hash(path): path
        for folder in followup_dirs
        for path in folder.iterdir()
        if path.is_file()
    }
    shared = baseline.keys() & followup.keys()
    print(
        f"Exact file overlap audit: {len(shared)} follow-up image files are byte-identical "
        f"to images in test_images/ ({len(followup)} follow-up files)."
    )

    if shared:
        baseline_rows = load_rows(ROOT / "eval_results_web.csv")
        baseline_csv = list(csv.DictReader((ROOT / "eval_results_web.csv").open(encoding="utf-8-sig", newline="")))
        unique_rows = []
        for row, parsed in zip(baseline_csv, baseline_rows):
            source = baseline_dir / row["file"]
            if source.is_file() and exact_hash(source) not in shared:
                unique_rows.append(parsed)
        print(f"Non-overlapping baseline subset: n={len(unique_rows)}")
        for threshold in thresholds:
            result = metrics(unique_rows, threshold)
            print(
                f"threshold={threshold:g} non-overlap confusion [ [TN={result['tn']}, FP={result['fp']}], "
                f"[FN={result['fn']}, TP={result['tp']}] ]; accuracy={result['accuracy']:.3f} "
                f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
                f"specificity={result['specificity']:.3f} F1={result['f1']:.3f} "
                f"FPR={result['fpr']:.3f} FNR={result['fnr']:.3f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args(argv)
    for path in args.files:
        rows = load_rows(path)
        print(f"\n{path.name}: n={len(rows)} (exploratory saved predictions)")
        for threshold in args.thresholds:
            result = metrics(rows, threshold)
            print(
                f"threshold={threshold:g} AI-positive confusion [ [TN={result['tn']}, FP={result['fp']}], "
                f"[FN={result['fn']}, TP={result['tp']}] ]; "
                f"accuracy={result['accuracy']:.3f} precision={result['precision']:.3f} "
                f"recall={result['recall']:.3f} specificity={result['specificity']:.3f} "
                f"F1={result['f1']:.3f} FPR={result['fpr']:.3f} FNR={result['fnr']:.3f}"
            )
    report_exact_overlap(args.thresholds)
    print("\nCaution: these development samples are not an untouched final evaluation set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
