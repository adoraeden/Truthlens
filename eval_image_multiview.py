"""Compare existing image-detector multi-view logits without loading a model.

The input is the exploratory ``preprocessing_comparison.csv`` ledger. The
output is a new per-image diagnostic CSV; this script never overwrites an
existing output or modifies production detector behavior.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "preprocessing_comparison.csv"
DEFAULT_OUTPUT = ROOT / "image_multiview_diagnostics.csv"
THRESHOLD = 2.0
VIEWS = {
    "q60": "A_current_q60",
    "q90": "C_resize_q90",
    "resize_only": "B_resize_only",
}


def threshold_relative_score(logit):
    """Return sigmoid(logit - cutoff), an uncalibrated score, not probability."""
    shifted = logit - THRESHOLD
    if shifted >= 0:
        return 1.0 / (1.0 + math.exp(-shifted))
    exp_shifted = math.exp(shifted)
    return exp_shifted / (1.0 + exp_shifted)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "folder", "actual"}
    required.update(f"{prefix}_logit" for prefix in VIEWS.values())
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing required multi-view columns")

    for row in rows:
        if row["actual"] not in {"0", "1"}:
            raise ValueError(f"Unknown label {row['actual']!r} for {row['filename']}")
        for prefix in VIEWS.values():
            logit = float(row[f"{prefix}_logit"])
            if not math.isfinite(logit):
                raise ValueError(f"Non-finite logit for {row['filename']} ({prefix})")
    return rows


def classify(row):
    logits = {
        name: float(row[f"{prefix}_logit"])
        for name, prefix in VIEWS.items()
    }
    flags = {name: logit >= THRESHOLD for name, logit in logits.items()}
    median_logit = statistics.median(logits.values())
    majority = sum(flags.values()) >= 2
    # Three-view median and 2-of-3 voting must agree for an odd view count.
    if (median_logit >= THRESHOLD) != majority:
        raise AssertionError("Median and majority predictions diverged")
    if all(flags.values()):
        consensus = "LIKELY AI-GENERATED"
    elif not any(flags.values()):
        consensus = "LIKELY REAL"
    else:
        consensus = "INCONCLUSIVE"
    return logits, flags, median_logit, majority, consensus


def metrics(rows, prediction):
    tp = fp = tn = fn = 0
    for row in rows:
        truth = int(row["actual"])
        predicted = bool(prediction(row))
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def print_metrics(name, result):
    print(
        f"{name}: n={result['n']} TP={result['tp']} FP={result['fp']} "
        f"TN={result['tn']} FN={result['fn']} "
        f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
        f"specificity={result['specificity']:.3f} accuracy={result['accuracy']:.3f} "
        f"FPR={result['fpr']:.3f} FNR={result['fnr']:.3f}"
    )


def write_diagnostics(rows, output):
    fields = ["filename", "folder", "actual"]
    for view in VIEWS:
        fields.extend((f"{view}_logit", f"{view}_score_uncalibrated", f"{view}_crosses_cutoff"))
    fields.extend((
        "median_logit", "median_prediction_ai", "majority_vote_ai",
        "unanimous_candidate_state",
    ))

    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            logits, flags, median_logit, majority, consensus = classify(row)
            result = {key: row[key] for key in ("filename", "folder", "actual")}
            for view in VIEWS:
                logit = logits[view]
                result[f"{view}_logit"] = f"{logit:.8f}"
                result[f"{view}_score_uncalibrated"] = f"{threshold_relative_score(logit):.8f}"
                result[f"{view}_crosses_cutoff"] = int(flags[view])
            result["median_logit"] = f"{median_logit:.8f}"
            result["median_prediction_ai"] = int(median_logit >= THRESHOLD)
            result["majority_vote_ai"] = int(majority)
            result["unanimous_candidate_state"] = consensus
            writer.writerow(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = read_rows(args.input)

    print(f"Input: {args.input.name}; n={len(rows)}; fixed logit cutoff={THRESHOLD:g}")
    print("Threshold-relative scores are sigmoid(logit - cutoff), not calibrated probabilities.")
    for view, prefix in VIEWS.items():
        print_metrics(
            f"single_{view}",
            metrics(rows, lambda row, p=prefix: float(row[f"{p}_logit"]) >= THRESHOLD),
        )
    print_metrics(
        "median_logit",
        metrics(rows, lambda row: classify(row)[2] >= THRESHOLD),
    )
    print_metrics(
        "majority_vote",
        metrics(rows, lambda row: classify(row)[3]),
    )
    consensus_result = metrics(
        rows,
        lambda row: classify(row)[4] == "LIKELY AI-GENERATED",
    )
    print_metrics("unanimous_AI_only (inconclusive is not an AI verdict)", consensus_result)

    states = {"LIKELY AI-GENERATED": 0, "LIKELY REAL": 0, "INCONCLUSIVE": 0}
    by_truth = {"0": dict.fromkeys(states, 0), "1": dict.fromkeys(states, 0)}
    for row in rows:
        state = classify(row)[4]
        states[state] += 1
        by_truth[row["actual"]][state] += 1
    print(f"Unanimous candidate states: {states}; by truth (0=real, 1=AI): {by_truth}")
    print("Candidate triage has no empirically selected near-boundary margin and is not production policy.")

    write_diagnostics(rows, args.output)
    print(f"Wrote new per-image/per-view diagnostics: {args.output}")
    print("Caution: these exploratory samples overlap the existing test_images evaluation set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
