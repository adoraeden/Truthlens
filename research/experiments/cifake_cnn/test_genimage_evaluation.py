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
