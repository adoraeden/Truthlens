
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


def run_evaluation(
    manifest_path,
    checkpoint_path,
    output_path,
    image_size=IMAGE_SIZE_MAIN,
    batch_size=BATCH_SIZE,
    device=None,
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

    with torch.no_grad():
        for images, labels, batch_generators in loader:
            images = images.to(selected_device)
            logits = model(images).reshape(-1)
            predicted = (torch.sigmoid(logits) >= 0.5).long()

            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
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
    args = parser.parse_args(argv)

    report = run_evaluation(
        args.manifest,
        args.checkpoint,
        args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()



========== research/experiments/cifake_cnn/test_genimage_evaluation.py ==========

import pandas as pd
import pytest
import torch
from PIL import Image

from research.experiments.cifake_cnn.evaluate_genimage import (
    GenImageManifestDataset,
    calculate_metrics,
)


def test_calculate_metrics_returns_expected_values():
    targets = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]

    metrics = calculate_metrics(targets, predictions)

    assert metrics["total_images"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_calculate_metrics_handles_perfect_predictions():
    metrics = calculate_metrics([0, 1, 0, 1], [0, 1, 0, 1])

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_manifest_dataset_loads_valid_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 80)).save(image_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "path": str(image_path),
        "generator": "TestGenerator",
        "class": "ai",
        "label": 1,
    }]).to_csv(manifest_path, index=False)

    dataset = GenImageManifestDataset(manifest_path, image_size=64)

    image, label, generator = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == 1
    assert generator == "TestGenerator"


def test_manifest_dataset_rejects_missing_columns(tmp_path):
    manifest_path = tmp_path / "invalid.csv"
    pd.DataFrame([{"path": "sample.jpg"}]).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError):
        GenImageManifestDataset(manifest_path, image_size=64)


def test_manifest_dataset_rejects_missing_image(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "path": str(tmp_path / "missing.jpg"),
        "generator": "TestGenerator",
        "class": "ai",
        "label": 1,
    }]).to_csv(manifest_path, index=False)

    with pytest.raises((FileNotFoundError, ValueError)):
        GenImageManifestDataset(manifest_path, image_size=64)
