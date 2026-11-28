"""Summarise a per-image prediction CSV written by evaluate_genimage.py."""

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "filename",
    "generator",
    "actual_label",
    "prob_fake",
    "confidence",
    "error_type",
}


def load_predictions(path):
    """Load a predictions CSV and check it has the columns we need."""
    frame = pd.read_csv(path, keep_default_na=False)

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Predictions file contains no rows")

    return frame


def summarise_errors(frame):
    """Count false positives and false negatives overall and per generator.

    False positive = real image predicted as AI.
    False negative = AI image predicted as real.
    """
    groups = [("ALL", frame)] + list(frame.groupby("generator"))

    rows = []
    for name, group in groups:
        real = int((group["actual_label"] == 0).sum())
        ai = int((group["actual_label"] == 1).sum())
        fp = int((group["error_type"] == "false_positive").sum())
        fn = int((group["error_type"] == "false_negative").sum())
        total = len(group)

        rows.append({
            "generator": name,
            "images": total,
            "accuracy": (total - fp - fn) / total,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fp / real if real else 0.0,
            "false_negative_rate": fn / ai if ai else 0.0,
        })

    return pd.DataFrame(rows)


def most_confident_errors(frame, error_type, top_n=10):
    """Return the mistakes of one type where the model was most sure."""
    errors = frame[frame["error_type"] == error_type]
    errors = errors.sort_values("confidence", ascending=False).head(top_n)
    return errors[["filename", "generator", "prob_fake", "confidence"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/experiments/cifake_cnn/outputs/genimage_error_summary.csv"
        ),
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)

    frame = load_predictions(args.predictions)
    summary = summarise_errors(frame)

    print("\nErrors by generator")
    print(summary.to_string(index=False))

    for error_type in ("false_positive", "false_negative"):
        print(f"\nMost confident {error_type} cases")
        print(most_confident_errors(frame, error_type, args.top_n).to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()