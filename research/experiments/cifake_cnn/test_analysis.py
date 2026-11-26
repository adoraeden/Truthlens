"""Unit tests for CIFAKE CNN training-history analysis."""

import csv

import pytest

from experiments.cifake_cnn.analysis import analyse_history, load_history


FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]


def write_history(path, rows, fieldnames=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows():
    return [
        {"epoch": 0, "train_loss": 0.25, "train_accuracy": 0.89, "val_loss": 0.19, "val_accuracy": 0.92},
        {"epoch": 1, "train_loss": 0.15, "train_accuracy": 0.94, "val_loss": 0.17, "val_accuracy": 0.93},
        {"epoch": 2, "train_loss": 0.06, "train_accuracy": 0.98, "val_loss": 0.12, "val_accuracy": 0.956},
        {"epoch": 3, "train_loss": 0.03, "train_accuracy": 0.99, "val_loss": 0.18, "val_accuracy": 0.95},
    ]


def test_load_history_reads_csv_metrics(tmp_path):
    path = tmp_path / "history.csv"
    write_history(path, sample_rows())

    rows = load_history(path)

    assert len(rows) == 4
    assert rows[2]["epoch"] == 2
    assert rows[2]["val_accuracy"] == pytest.approx(0.956)


def test_load_history_rejects_missing_columns(tmp_path):
    path = tmp_path / "history.csv"
    write_history(path, [{"epoch": 0}], fieldnames=["epoch"])

    with pytest.raises(ValueError, match="missing required columns"):
        load_history(path)


def test_load_history_rejects_invalid_numeric_values(tmp_path):
    path = tmp_path / "history.csv"
    rows = sample_rows()
    rows[1]["val_loss"] = "not-a-number"
    write_history(path, rows)

    with pytest.raises(ValueError, match="invalid metric"):
        load_history(path)


def test_load_history_rejects_non_finite_values(tmp_path):
    path = tmp_path / "history.csv"
    rows = sample_rows()
    rows[1]["val_loss"] = "nan"
    write_history(path, rows)

    with pytest.raises(ValueError, match="non-finite"):
        load_history(path)


def test_load_history_rejects_empty_history(tmp_path):
    path = tmp_path / "history.csv"
    write_history(path, [])

    with pytest.raises(ValueError, match="contains no epochs"):
        load_history(path)


def test_analysis_selects_epoch_with_lowest_validation_loss():
    summary = analyse_history(sample_rows())

    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == pytest.approx(0.12)
    assert summary["best_val_accuracy"] == pytest.approx(0.956)


def test_analysis_calculates_accuracy_and_loss_gaps():
    summary = analyse_history(sample_rows())

    assert summary["best_epoch_accuracy_gap"] == pytest.approx(0.98 - 0.956)
    assert summary["best_epoch_loss_gap"] == pytest.approx(0.12 - 0.06)


def test_analysis_flags_possible_overfitting_after_best_epoch():
    summary = analyse_history(sample_rows())

    assert summary["validation_loss_increased_after_best"] is True
    assert summary["possible_overfitting"] is True


def test_analysis_does_not_flag_overfitting_without_later_validation_increase():
    rows = sample_rows()[:3]
    summary = analyse_history(rows)

    assert summary["validation_loss_increased_after_best"] is False
    assert summary["possible_overfitting"] is False


def test_analysis_rejects_empty_rows():
    with pytest.raises(ValueError, match="contains no epochs"):
        analyse_history([])
