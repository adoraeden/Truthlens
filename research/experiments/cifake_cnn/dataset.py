"""CIFAKE loading and split utilities for the from-scratch CNN experiment.

Dataset root: the local CIFAKE download at ``research/data/kaggle_image/`` (git-ignored).
Only ``train/`` is used for training/validation; the official ``test/``
directory stays separate for final evaluation.

Label convention (explicit, not the alphabetical default):
    FAKE = 1  (positive class)
    REAL = 0
"""

import random
from collections import Counter
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .config import (
    IMAGE_SIZE_MAIN,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RANDOM_SEED,
    VAL_FRACTION,
)

CIFAKE_ROOT = Path(__file__).resolve().parents[2] / "data" / "kaggle_image"


def _transform(image_size):
    """Resize to a square, convert to tensor, and normalize with ImageNet stats."""
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_datasets(image_size=IMAGE_SIZE_MAIN):
    """Load CIFAKE train/test via ImageFolder with explicit FAKE=1, REAL=0 labels."""
    transform = _transform(image_size)
    train_ds = ImageFolder(CIFAKE_ROOT / "train", transform=transform)
    test_ds = ImageFolder(CIFAKE_ROOT / "test", transform=transform)
    positive_index = train_ds.class_to_idx[POSITIVE_CLASS]
    for ds in (train_ds, test_ds):
        # ImageFolder sorts class names alphabetically: FAKE=0, REAL=1.
        # Remap so the positive class is always 1 (experiment convention:
        # AI-generated is the positive class).
        ds.samples = [
            (path, 1 if target == positive_index else 0)
            for path, target in ds.samples
        ]
        if hasattr(ds, "targets"):
            ds.targets = [label for _, label in ds.samples]
    return train_ds, test_ds


def make_train_val_split(dataset, seed=RANDOM_SEED, val_fraction=VAL_FRACTION):
    """Reproducible stratified 90/10 split of the CIFAKE training directory.

    Stratification keeps the class proportions balanced: the CIFAKE train
    directory holds 50,000 REAL and 50,000 FAKE, so the 10% validation split
    is exactly 5,000 REAL and 5,000 FAKE (train = 45,000 of each).

    Returns ``(train_indices, val_indices)``. Deterministic given ``seed``.
    """
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    val_indices = []
    for label in sorted(class_indices):
        indices = class_indices[label]
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        val_indices.extend(indices[:n_val])
    val_indices = sorted(val_indices)
    val_set = set(val_indices)
    train_indices = [idx for idx in range(len(dataset)) if idx not in val_set]
    return train_indices, val_indices


def class_counts(dataset):
    """Return {label: count} over the (remapped) labels of an ImageFolder dataset."""
    return dict(sorted(Counter(label for _, label in dataset.samples).items()))


def verify_label_mapping(train_ds, test_ds):
    """Assert every REAL file has label 0 and every FAKE file has label 1."""
    for ds in (train_ds, test_ds):
        for path, label in ds.samples:
            expected = 1 if Path(path).parent.name == POSITIVE_CLASS else 0
            assert label == expected, f"label mismatch: {path} -> {label} (expected {expected})"
    print(f"label mapping OK: {POSITIVE_CLASS}=1, {NEGATIVE_CLASS}=0")


def verify_split(dataset, train_indices, val_indices):
    """Assert non-empty, disjoint, and reproducible train/val indices."""
    assert train_indices and val_indices, "both splits must be non-empty"
    assert not set(train_indices) & set(val_indices), "train/val indices overlap"
    assert set(train_indices) | set(val_indices) == set(range(len(dataset)))
    again_train, again_val = make_train_val_split(dataset)
    assert again_train == train_indices and again_val == val_indices
    print(f"split OK: {len(train_indices)} train / {len(val_indices)} val (seed {RANDOM_SEED}); "
          f"disjoint and reproducible")
