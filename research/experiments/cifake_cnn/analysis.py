"""Analyse CIFAKE CNN training history exported by training.py.

Usage:
    python -m experiments.cifake_cnn.analysis path/to/training_history.csv
"""

import argparse
import csv
import json
from pathlib import Path

REQUIRED_COLUMNS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)


def load_history(csv_path):
    """Load and validate epoch metrics from a training-history CSV."""
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_loss": float(row["val_loss"]),
                    "val_accuracy": float(row["val_accuracy"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid metric on CSV line {line_number}") from exc

            for key in REQUIRED_COLUMNS[1:]:
                value = parsed[key]
                if not __import__("math").isfinite(value):
                    raise ValueError(f"non-finite {key} on CSV line {line_number}")
            rows.append(parsed)

    if not rows:
        raise ValueError("training history contains no epochs")
    return rows


def analyse_history(rows):
    """Return best-epoch metrics, generalisation gaps, and overfitting signal."""
    if not rows:
        raise ValueError("training history contains no epochs")

    best = min(rows, key=lambda row: row["val_loss"])
    final = rows[-1]
    best_index = rows.index(best)
    later_rows = rows[best_index + 1 :]
    validation_loss_increased = bool(later_rows) and all(
        row["val_loss"] > best["val_loss"] for row in later_rows
    )

    return {
        "epochs_completed": len(rows),
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
        "best_train_accuracy": best["train_accuracy"],
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "final_epoch": final["epoch"],
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_val_loss": final["val_loss"],
        "final_val_accuracy": final["val_accuracy"],
        "best_epoch_accuracy_gap": best["train_accuracy"] - best["val_accuracy"],
        "best_epoch_loss_gap": best["val_loss"] - best["train_loss"],
        "validation_loss_increased_after_best": validation_loss_increased,
        "possible_overfitting": validation_loss_increased
        and final["train_loss"] < best["train_loss"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", type=Path, help="Path to training_history.csv")
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON summary")
    args = parser.parse_args(argv)

    summary = analyse_history(load_history(args.history_csv))
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
