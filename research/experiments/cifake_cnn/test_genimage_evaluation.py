import pandas as pd
import pytest
import torch
from PIL import Image

from research.experiments.cifake_cnn.evaluate_genimage import (
    GenImageManifestDataset,
    build_prediction_report,
    calculate_metrics,
    run_evaluation,
)
from research.experiments.cifake_cnn.model import CifakeCNN


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



def test_build_prediction_report_marks_errors_correctly():
    frame = pd.DataFrame({
        "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "generator": ["G1", "G1", "G2", "G2"],
        "class": ["real", "real", "ai", "ai"],
        "label": [0, 0, 1, 1],
    })

    report = build_prediction_report(frame, [0.1, 0.8, 0.9, 0.3])

    assert report["predicted_label"].tolist() == [0, 1, 1, 0]
    assert report["predicted_class"].tolist() == ["real", "ai", "ai", "real"]
    assert report["correct"].tolist() == [True, False, True, False]
    assert report["error_type"].tolist() == [
        "none", "false_positive", "none", "false_negative"
    ]
    assert report["confidence"].tolist() == pytest.approx([0.9, 0.8, 0.9, 0.7])
    assert report["filename"].tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]


def test_build_prediction_report_treats_threshold_as_fake():
    frame = pd.DataFrame({
        "path": ["a.jpg"], "generator": ["G1"], "class": ["real"], "label": [0],
    })

    report = build_prediction_report(frame, [0.5])

    assert report["predicted_label"].tolist() == [1]
    assert report["error_type"].tolist() == ["false_positive"]


def test_build_prediction_report_rejects_length_mismatch():
    frame = pd.DataFrame({
        "path": ["a.jpg"], "generator": ["G1"], "class": ["real"], "label": [0],
    })

    with pytest.raises(ValueError):
        build_prediction_report(frame, [0.1, 0.9])



def test_run_evaluation_saves_prediction_csv(tmp_path):
    rows = []
    for i, cls in enumerate(["real", "ai", "real", "ai"]):
        image_path = tmp_path / f"img{i}.png"
        Image.new("RGB", (64, 64), color=(i * 40, 80, 120)).save(image_path)
        rows.append({
            "path": str(image_path),
            "generator": "TestGenerator",
            "class": cls,
            "label": 0 if cls == "real" else 1,
        })

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    model = CifakeCNN(input_size=64)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"epoch": 1, "model_state_dict": model.state_dict()}, checkpoint_path
    )

    predictions_path = tmp_path / "predictions.csv"
    run_evaluation(
        manifest_path,
        checkpoint_path,
        tmp_path / "results.json",
        image_size=64,
        batch_size=2,
        device="cpu",
        predictions_path=predictions_path,
    )

    report = pd.read_csv(predictions_path, keep_default_na=False)

    assert len(report) == 4
    assert report["filename"].tolist() == [f"img{i}.png" for i in range(4)]
    assert set(report["error_type"]) <= {
        "none", "false_positive", "false_negative"
    }
    assert int(report["correct"].sum()) == int((report["error_type"] == "none").sum())