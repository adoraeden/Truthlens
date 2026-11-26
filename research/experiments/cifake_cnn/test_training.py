"""Tests for the CIFAKE CNN training utilities (research/experiments/cifake_cnn/training.py).

Uses small synthetic datasets and models only — no full CIFAKE training.
"""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import training
from .model import CifakeCNN


class _FakeDataset:
    """Minimal ImageFolder-like dataset for DataLoader/split utilities."""

    def __init__(self, n=40):
        self.samples = [(f"item_{i}", i % 2) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        torch.manual_seed(1234 + index)
        inputs = torch.randn(3, 8, 8)
        return inputs, self.samples[index][1]


class TrainingUtilityTests(unittest.TestCase):
    """Tests for device selection, seeding, loaders, metrics, early stopping."""

    def test_select_device_returns_torch_device(self):
        device = training.select_device()
        self.assertIsInstance(device, torch.device)

    def test_set_seed_sets_torch_seed(self):
        training.set_seed(1234)
        self.assertEqual(torch.initial_seed(), 1234)
        training.set_seed(4321)
        self.assertEqual(torch.initial_seed(), 4321)

    def test_make_loaders_splits_are_disjoint_and_cover_all(self):
        dataset = _FakeDataset(40)
        train_loader, val_loader = training.make_loaders(dataset, batch_size=8, val_fraction=0.25)
        train_idx = set(train_loader.dataset.indices)
        val_idx = set(val_loader.dataset.indices)
        self.assertFalse(train_idx & val_idx, "train and val must be disjoint")
        self.assertEqual(train_idx | val_idx, set(range(40)), "indices must cover the dataset")
        self.assertEqual(len(train_idx) + len(val_idx), 40)

    def test_train_epoch_returns_finite_loss_and_accuracy(self):
        model = CifakeCNN(widths=(4,), head_width=4)
        loader = DataLoader(_FakeDataset(8), batch_size=4)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters())
        loss, accuracy = training.train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_evaluate_returns_loss_and_accuracy_in_range(self):
        model = CifakeCNN(widths=(4,), head_width=4)
        loader = DataLoader(_FakeDataset(8), batch_size=4)
        loss, accuracy = training.evaluate(model, loader, torch.nn.BCEWithLogitsLoss(), torch.device("cpu"))
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_early_stopping_stops_after_patience(self):
        early = training.EarlyStopping(patience=2)
        self.assertFalse(early.update(1.0))   # first loss -> best
        self.assertFalse(early.update(0.9))   # improvement
        self.assertFalse(early.update(0.95))  # bad epoch 1
        self.assertTrue(early.update(0.96))   # bad epoch 2 >= patience

    def test_save_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints" / "best_cifake_cnn.pt"
            model = CifakeCNN(widths=(4,), head_width=4)
            optimizer = torch.optim.AdamW(model.parameters())
            saved = training.save_checkpoint(model, optimizer, epoch=3, val_loss=0.25,
                                              val_accuracy=0.6, path=path,
                                              train_loss=0.4, train_accuracy=0.5,
                                              seed=7, output_dir=tmp)
            self.assertTrue(saved.is_file())
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["val_loss"], 0.25)
            self.assertEqual(checkpoint["best_val_loss"], 0.25)
            self.assertEqual(checkpoint["train_loss"], 0.4)
            self.assertEqual(checkpoint["train_accuracy"], 0.5)
            self.assertEqual(checkpoint["seed"], 7)
            self.assertEqual(checkpoint["output_dir"], tmp)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)

    def test_load_checkpoint_restores_model_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints" / "best_cifake_cnn.pt"
            model = CifakeCNN(widths=(4,), head_width=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loader = DataLoader(_FakeDataset(8), batch_size=4)
            criterion = torch.nn.BCEWithLogitsLoss()
            train_loss, train_accuracy = training.train_epoch(
                model, loader, criterion, optimizer, torch.device("cpu"))
            saved_state = {k: v.clone() for k, v in model.state_dict().items()}
            saved_opt_state = optimizer.state_dict()
            self.assertGreater(len(saved_opt_state.get("state", {})), 0,
                               "expected optimizer state after a training step")
            training.save_checkpoint(model, optimizer, epoch=7, val_loss=0.3, val_accuracy=0.7,
                                     train_loss=train_loss, train_accuracy=train_accuracy,
                                     seed=42, output_dir=tmp, path=path)
            # Fresh model/optimizer; load_checkpoint must restore weights and state.
            fresh_model = CifakeCNN(widths=(4,), head_width=4)
            fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
            info = training.load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer,
                                            device=torch.device("cpu"))
            # Metadata returned
            self.assertEqual(info["epoch"], 7)
            self.assertAlmostEqual(info["val_loss"], 0.3, places=6)
            self.assertAlmostEqual(info["val_accuracy"], 0.7, places=6)
            self.assertIsNotNone(info["train_loss"])
            self.assertIsNotNone(info["train_accuracy"])
            self.assertEqual(info["seed"], 42)
            # Model weights restored
            for key, value in saved_state.items():
                self.assertTrue(torch.equal(fresh_model.state_dict()[key], value),
                                f"weight mismatch: {key}")
            # Optimizer state restored
            loaded_opt = fresh_optimizer.state_dict()
            self.assertEqual(len(loaded_opt["state"]), len(saved_opt_state["state"]))

    def test_load_checkpoint_missing_fields_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.pt"
            model = CifakeCNN(widths=(4,), head_width=4)
            torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, path)
            loaded = training.load_checkpoint(path)
            self.assertEqual(loaded["epoch"], 3)
            self.assertIsNone(loaded["val_loss"])
            self.assertIsNone(loaded["train_loss"])
            self.assertIsNone(loaded["train_accuracy"])
            self.assertIsNone(loaded["seed"])
            self.assertIsNone(loaded["output_dir"])

    def test_append_history_row_preserves_earlier_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.csv"
            row0 = {"epoch": 0, "train_loss": 1.0, "train_accuracy": 0.5,
                    "val_loss": 1.1, "val_accuracy": 0.5}
            row1 = {"epoch": 1, "train_loss": 0.9, "train_accuracy": 0.6,
                    "val_loss": 1.2, "val_accuracy": 0.4}
            training.append_history_row(history_file, row0)
            training.append_history_row(history_file, row1)
            with history_file.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([int(r["epoch"]) for r in rows], [0, 1])
            training.append_history_row(history_file, row0)  # re-append keeps prior rows
            with history_file.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)

    def test_run_training_smoke_saves_history_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _FakeDataset(20)
            model = CifakeCNN(widths=(4,), head_width=4)
            train_loader, val_loader = training.make_loaders(dataset, batch_size=4, val_fraction=0.2)
            history = training.run_training(
                model, train_loader, val_loader, torch.device("cpu"),
                epochs=2, patience=2,
                checkpoint_dir=Path(tmp) / "checkpoints",
                history_file=Path(tmp) / "history.csv",
                verbose=False,
            )
            self.assertEqual(len(history), 2)
            self.assertTrue((Path(tmp) / "history.csv").is_file())
            checkpoints = list((Path(tmp) / "checkpoints").glob("*.pt"))
            self.assertGreaterEqual(len(checkpoints), 1)
            with (Path(tmp) / "history.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
