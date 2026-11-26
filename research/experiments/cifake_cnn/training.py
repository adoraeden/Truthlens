"""Training pipeline for the from-scratch CIFAKE CNN.

Relies on the existing experiment package:

- ``CifakeCNN`` and ``count_parameters`` from ``model.py``
- ``make_datasets``, ``make_train_val_split`` from ``dataset.py``
- hyperparameters from ``config.py``

Usage (from the repo root)::

    python -B -m experiments.cifake_cnn.training [--device cuda] [--output-dir ...]

Outputs are isolated under ``experiments/cifake_cnn/outputs/`` by default:

- ``outputs/checkpoints/best_cifake_cnn.pt`` — best-epoch checkpoint
  (model state, optimizer state, epoch, train/val metrics, best_val_loss,
  seed, output_dir). Restorable via :func:`load_checkpoint`.
- ``outputs/history.csv`` — per-epoch train/val loss and accuracy (append-only).
"""

import argparse
import csv
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE_MAIN,
    LEARNING_RATE,
    MAX_EPOCHS,
    RANDOM_SEED,
    VAL_FRACTION,
    WEIGHT_DECAY,
)
from .dataset import make_datasets, make_train_val_split
from .model import CifakeCNN

# --- Output locations (configurable via arguments) ---
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_HISTORY_FILE = DEFAULT_OUTPUT_DIR / "history.csv"
BEST_CHECKPOINT_NAME = "best_cifake_cnn.pt"


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    """Return ``cuda`` when available, otherwise ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(dataset, seed=RANDOM_SEED, val_fraction=VAL_FRACTION,
                 batch_size=BATCH_SIZE, shuffle_train=True):
    """Build train/validation DataLoaders via the existing reproducible split.

    Uses ``make_train_val_split`` from ``dataset.py``; the caller's ``dataset``
    is split into disjoint ``Subset``s (train + val). Returns
    ``(train_loader, val_loader)``.
    """
    train_indices, val_indices = make_train_val_split(
        dataset, seed=seed, val_fraction=val_fraction)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch; return ``(avg_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        logits = model(inputs).reshape(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * inputs.shape[0]
        with torch.no_grad():
            preds = (torch.sigmoid(logits.detach()) >= 0.5).long()
            correct += int((preds == targets.long()).sum().item())
        total += inputs.shape[0]
    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Return ``(avg_loss, accuracy)`` on the loader without updating weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            logits = model(inputs).reshape(-1)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * inputs.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += int((preds == targets.long()).sum().item())
            total += inputs.shape[0]
    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


class EarlyStopping:
    """Small early-stopping helper based on validation loss."""

    def __init__(self, patience, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_val_loss = None
        self.bad_epochs = 0

    def update(self, val_loss):
        """Record ``val_loss``; return True when training should stop."""
        if self.best_val_loss is None or val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, path,
                    train_loss=None, train_accuracy=None, seed=None, output_dir=None):
    """Save a best-epoch checkpoint (model, optimizer, epoch, metadata).

    Backward compatible with earlier callers: the extra arguments are optional
    and stored as ``None`` when not supplied.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "best_val_loss": float(val_loss),
        # --- metadata (added in the restoration stage) ---
        "train_loss": float(train_loss) if train_loss is not None else None,
        "train_accuracy": float(train_accuracy) if train_accuracy is not None else None,
        "seed": int(seed) if seed is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model=None, optimizer=None, device=None):
    """Load a checkpoint created by :func:`save_checkpoint`.

    - Restores model weights via ``model.load_state_dict``.
    - Restores optimizer state via ``optimizer.load_state_dict`` when an
      optimizer is supplied (and the checkpoint has optimizer state).
    - Loads tensors onto ``device`` via ``torch.load(map_location=device)``;
      ``None`` uses the default (checkpoint's saved device).

    Returns a dict of available metadata: ``epoch``, ``val_loss``,
    ``val_accuracy``, ``train_loss``, ``train_accuracy``, ``seed``,
    ``output_dir`` and ``best_val_loss`` (``None`` when absent, so
    checkpoints written before the metadata stage still load safely).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    if model is not None and checkpoint.get("model_state_dict") is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "val_accuracy": checkpoint.get("val_accuracy"),
        "train_loss": checkpoint.get("train_loss"),
        "train_accuracy": checkpoint.get("train_accuracy"),
        "seed": checkpoint.get("seed"),
        "output_dir": checkpoint.get("output_dir"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }


def append_history_row(history_file, row):
    """Append one history row to the CSV; existing rows are preserved."""
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


def run_training(model, train_loader, val_loader, device,
                 epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
                 seed=RANDOM_SEED, checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                 history_file=DEFAULT_HISTORY_FILE, verbose=True):
    """Run the full training loop and return the history rows.

    - Uses ``BCEWithLogitsLoss`` (model outputs raw logits).
    - Optimizer: ``AdamW`` with ``LEARNING_RATE`` and ``WEIGHT_DECAY``.
    - Tracks train/val loss and accuracy per epoch; appends to ``history_file``.
    - Saves the best checkpoint (by val loss) to ``checkpoint_dir``.
    - Early stopping with ``patience`` (from config when not overridden).
    """
    set_seed(seed)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience)
    history = []
    best_val_loss = math.inf
    best_checkpoint_path = checkpoint_dir / BEST_CHECKPOINT_NAME

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 4),
        }
        history.append(row)
        if history_file is not None:
            append_history_row(history_file, row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_accuracy, best_checkpoint_path,
                train_loss=train_loss, train_accuracy=train_accuracy,
                seed=seed, output_dir=checkpoint_dir.parent,
            )
            if verbose:
                print(f"epoch {epoch}: new best val_loss={val_loss:.4f} -> {best_checkpoint_path}")
        if early_stopping.update(val_loss):
            if verbose:
                print(f"early stopping at epoch {epoch} (patience={patience})")
            break

    if verbose and history:
        last = history[-1]
        print(f"finished: {len(history)} epochs; last val_loss={last['val_loss']:.4f}, "
              f"last val_accuracy={last['val_accuracy']:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the from-scratch CIFAKE CNN.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_MAIN,
                        help="Square input size (default: IMAGE_SIZE_MAIN)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Training batch size (default: BATCH_SIZE)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help="Maximum epochs (default: MAX_EPOCHS)")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu'; default: auto-select")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed (default: RANDOM_SEED)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for checkpoints/history (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device() if args.device is None else torch.device(args.device)

    train_ds, _test_ds = make_datasets(image_size=args.image_size)
    model = CifakeCNN(input_size=args.image_size)
    train_loader, val_loader = make_loaders(train_ds, batch_size=args.batch_size)

    history = run_training(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_dir=args.output_dir / "checkpoints",
        history_file=args.output_dir / "history.csv",
    )
    print(f"outputs: {args.output_dir}")
    if history:
        print(f"best val_loss: {min(row['val_loss'] for row in history):.4f}")


if __name__ == "__main__":
    main()
