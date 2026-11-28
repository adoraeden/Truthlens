
"""Evaluate a trained CIFAKE CNN on a labelled GenImage manifest."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import IMAGE_SIZE_MAIN, BATCH_SIZE
from .dataset import _deterministic_transform
from .model import CifakeCNN
from .training import load_checkpoint, select_device


class GenImageManifestDataset(Dataset):
    """Load images and labels from a CSV manifest."""

    def __init__(self, manifest_path, image_size=IMAGE_SIZE_MAIN):
        self.frame = pd.read_csv(manifest_path)
        required = {"path", "generator", "class", "label"}

        if not required.issubset(self.frame.columns):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        if self.frame.empty:
            raise ValueError("Manifest contains no images")

        self.frame["label"] = self.frame["label"].astype(int)

        if not self.frame["label"].isin([0, 1]).all():
            raise ValueError("Labels must be 0 (REAL) or 1 (FAKE)")

        expected = self.frame["class"].map({"real": 0, "ai": 1})
        if expected.isna().any() or not (expected == self.frame["label"]).all():
            raise ValueError("Class names and numeric labels do not match")

        missing = [p for p in self.frame["path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest image paths are missing. "
                f"First missing path: {missing[0]}"
            )

        self.transform = _deterministic_transform(image_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, int(row["label"]), str(row["generator"])


def calculate_metrics(targets, predictions):
    """Calculate binary metrics with REAL=0 and FAKE=1."""
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "total_images": len(targets),
        "accuracy": (tp + tn) / len(targets),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "class_mapping": {"negative": "REAL", "positive": "FAKE"},
    }


def build_prediction_report(frame, probabilities, threshold=0.5):
    """Build a per-image prediction table (REAL=0, FAKE=1).

    ``probabilities`` holds the model's FAKE probability for each manifest
    row, in the same order as ``frame``.
    """
    if len(frame) != len(probabilities):
        raise ValueError(
            f"Got {len(probabilities)} probabilities for {len(frame)} images"
        )

    frame = frame.reset_index(drop=True)
    probs = [float(p) for p in probabilities]
    predicted = [int(p >= threshold) for p in probs]
    actual = frame["label"].astype(int).tolist()

    error_types = []
    for a, p in zip(actual, predicted):
        if a == 0 and p == 1:
            error_types.append("false_positive")
        elif a == 1 and p == 0:
            error_types.append("false_negative")
        else:
            error_types.append("none")

    return pd.DataFrame({
        "path": frame["path"],
        "filename": [Path(p).name for p in frame["path"]],
        "generator": frame["generator"],
        "actual_class": frame["class"],
        "actual_label": actual,
        "predicted_class": ["ai" if p == 1 else "real" for p in predicted],
        "predicted_label": predicted,
        "prob_fake": probs,
        "confidence": [p if pred == 1 else 1 - p
                       for p, pred in zip(probs, predicted)],
        "correct": [a == p for a, p in zip(actual, predicted)],
        "error_type": error_types,
    })

def run_evaluation(
    manifest_path,
    checkpoint_path,
    output_path,
    image_size=IMAGE_SIZE_MAIN,
    batch_size=BATCH_SIZE,
    device=None,
    predictions_path=None,
):
    selected_device = select_device() if device is None else torch.device(device)

    dataset = GenImageManifestDataset(manifest_path, image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = CifakeCNN(input_size=image_size)
    checkpoint_info = load_checkpoint(
        checkpoint_path, model=model, device=selected_device
    )
    model.to(selected_device)
    model.eval()

    targets = []
    predictions = []
    generators = []
    probabilities = []

    with torch.no_grad():
        for images, labels, batch_generators in loader:
            images = images.to(selected_device)
            logits = model(images).reshape(-1)
            batch_probs = torch.sigmoid(logits)
            predicted = (batch_probs >= 0.5).long()

            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            probabilities.extend(batch_probs.cpu().tolist())
            generators.extend(batch_generators)

    overall = calculate_metrics(targets, predictions)

    per_generator = {}
    for generator in sorted(set(generators)):
        indices = [
            i for i, name in enumerate(generators) if name == generator
        ]
        per_generator[generator] = calculate_metrics(
            [targets[i] for i in indices],
            [predictions[i] for i in indices],
        )

    report = {
        "dataset": "GenImage manifest evaluation",
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_info["epoch"],
        "threshold": 0.5,
        "image_size": image_size,
        "device": str(selected_device),
        "overall": overall,
        "by_generator": per_generator,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        build_prediction_report(dataset.frame, probabilities).to_csv(
            predictions_path, index=False
        )
        
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/experiments/cifake_cnn/outputs/genimage_results.json"
        ),
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_MAIN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path(
            "research/experiments/cifake_cnn/outputs/genimage_predictions.csv"
        ),
    )
    args = parser.parse_args(argv)

    report = run_evaluation(
        args.manifest,
        args.checkpoint,
        args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        predictions_path=args.predictions_output,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
