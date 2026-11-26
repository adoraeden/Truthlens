"""Tests for official CIFAKE checkpoint evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from . import evaluate, training
from .model import CifakeCNN


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits[: inputs.shape[0]].reshape(-1, 1)


class EvaluationTests(unittest.TestCase):
    def test_calculate_metrics_returns_confusion_matrix_and_scores(self):
        result = evaluate.calculate_metrics(
            [0, 0, 1, 1], [0, 1, 0, 1]
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1_score"], 0.5)
        self.assertEqual(result["total_images"], 4)

    def test_calculate_metrics_handles_no_positive_predictions(self):
        result = evaluate.calculate_metrics([0, 1], [0, 0])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1_score"], 0.0)

    def test_calculate_metrics_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate.calculate_metrics([], [])

    def test_calculate_metrics_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            evaluate.calculate_metrics([0, 1], [1])

    def test_calculate_metrics_rejects_invalid_labels(self):
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            evaluate.calculate_metrics([0, 2], [0, 1])

    def test_evaluate_model_runs_inference_and_reports_test_loss(self):
        inputs = torch.zeros(4, 3, 2, 2)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        model = _FixedLogitModel([-2.0, 2.0, -2.0, 2.0])

        result = evaluate.evaluate_model(model, loader, torch.device("cpu"))

        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["total_images"], 4)
        self.assertGreater(result["test_loss"], 0.0)
        self.assertFalse(model.training)

    def test_evaluate_model_rejects_empty_loader(self):
        loader = DataLoader(
            TensorDataset(torch.empty(0, 3, 2, 2), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate.evaluate_model(_FixedLogitModel([]), loader, torch.device("cpu"))

    def test_run_evaluation_restores_checkpoint_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "best.pt"
            output_path = root / "report.json"
            model = CifakeCNN(input_size=32)
            optimizer = torch.optim.AdamW(model.parameters())
            training.save_checkpoint(
                model, optimizer, epoch=2, val_loss=0.2, val_accuracy=0.9,
                path=checkpoint_path,
            )
            fake_dataset = TensorDataset(
                torch.zeros(4, 3, 32, 32),
                torch.tensor([0, 0, 1, 1]),
            )

            with patch.object(evaluate, "make_datasets", return_value=(None, fake_dataset)):
                report = evaluate.run_evaluation(
                    checkpoint_path, output_path, image_size=32,
                    batch_size=2, device="cpu",
                )

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["checkpoint_epoch"], 2)
            self.assertEqual(report["dataset"], "CIFAKE official test split")
            self.assertEqual(saved["metrics"]["total_images"], 4)
            self.assertEqual(saved["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
