import pandas as pd
import pytest

from research.experiments.cifake_cnn.analyze_predictions import (
    load_predictions,
    most_confident_errors,
    summarise_errors,
)


def make_frame():
    columns = [
        "filename", "generator", "actual_label", "predicted_label",
        "prob_fake", "confidence", "error_type",
    ]
    rows = [
        ("a.png", "G1", 0, 0, 0.1, 0.9, "none"),
        ("b.png", "G1", 0, 1, 0.8, 0.8, "false_positive"),
        ("c.png", "G2", 1, 1, 0.9, 0.9, "none"),
        ("d.png", "G2", 1, 0, 0.2, 0.8, "false_negative"),
        ("e.png", "G2", 1, 0, 0.4, 0.6, "false_negative"),
    ]
    return pd.DataFrame(rows, columns=columns)


def test_summarise_errors_counts_per_generator():
    summary = summarise_errors(make_frame()).set_index("generator")

    assert summary.loc["ALL", "images"] == 5
    assert summary.loc["ALL", "accuracy"] == pytest.approx(0.4)
    assert summary.loc["ALL", "false_positives"] == 1
    assert summary.loc["ALL", "false_negatives"] == 2

    assert summary.loc["G1", "false_positives"] == 1
    assert summary.loc["G1", "false_positive_rate"] == pytest.approx(0.5)
    assert summary.loc["G1", "false_negative_rate"] == 0.0

    assert summary.loc["G2", "false_negatives"] == 2
    assert summary.loc["G2", "false_negative_rate"] == pytest.approx(2 / 3)
    assert summary.loc["G2", "accuracy"] == pytest.approx(1 / 3)


def test_most_confident_errors_sorts_by_confidence():
    frame = make_frame()

    negatives = most_confident_errors(frame, "false_negative")
    assert negatives["filename"].tolist() == ["d.png", "e.png"]

    positives = most_confident_errors(frame, "false_positive", top_n=1)
    assert positives["filename"].tolist() == ["b.png"]


def test_load_predictions_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"filename": "a.png"}]).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_predictions(path)