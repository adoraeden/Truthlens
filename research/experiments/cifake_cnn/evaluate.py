"""Evaluate a trained CIFAKE CNN checkpoint on the official test split.

Run from the repository root:
    python -m experiments.cifake_cnn.evaluate --checkpoint PATH_TO_CHECKPOINT
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import IMAGE_SIZE_MAIN, NEGATIVE_CLASS, POSITIVE_CLASS, BATCH_SIZE
from .dataset import make_datasets
from .model import CifakeCNN
from .training import load_checkpoint, select_device


def calculate_metrics(targets, predictions):
    """Calculate binary metrics with FAKE=1 and REAL=0."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("cannot evaluate an empty set")
    if any(value not in (0, 1) for value in targets + predictions):
        raise ValueError("targets and predictions must contain only 0 or 1")

    true_negative = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    true_positive = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    total = len(targets)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "total_images": total,
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "class_mapping": {"negative": NEGATIVE_CLASS, "positive": POSITIVE_CLASS},
    }


def evaluate_model(model, loader, device):
    """Run inference and return metrics plus mean BCE-with-logits test loss."""
    model.to(device)
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    targets = []
    predictions = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            logits = model(inputs).reshape(-1)
            total_loss += float(criterion(logits, labels.float()).item())
            probabilities = torch.sigmoid(logits)
            batch_predictions = (probabilities >= 0.5).long()
            targets.extend(labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            total += labels.numel()

    metrics = calculate_metrics(targets, predictions)
    metrics["test_loss"] = total_loss / total
    return metrics


def run_evaluation(checkpoint_path, output_path, image_size=IMAGE_SIZE_MAIN,
                   batch_size=BATCH_SIZE, device=None):
    """Load the official CIFAKE test split, restore a checkpoint, and save JSON."""
    selected_device = select_device() if device is None else torch.device(device)
    _, test_dataset = make_datasets(image_size=image_size)
    if len(test_dataset) == 0:
        raise ValueError("official CIFAKE test dataset is empty")

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = CifakeCNN(input_size=image_size)
    checkpoint_info = load_checkpoint(
        checkpoint_path, model=model, device=selected_device
    )
    metrics = evaluate_model(model, loader, selected_device)
    report = {
        "dataset": "CIFAKE official test split",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_info["epoch"],
        "threshold": 0.5,
        "image_size": image_size,
        "device": str(selected_device),
        "metrics": metrics,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained best_cifake_cnn.pt checkpoint")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/cifake_cnn/outputs/official_test_results.json"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_MAIN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'; default: auto-select")
    args = parser.parse_args(argv)

    report = run_evaluation(
        args.checkpoint, args.output, image_size=args.image_size,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
