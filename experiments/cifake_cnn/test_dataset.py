"""Tests for CIFAKE dataset loading, labels, and train/validation splitting."""

from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.cifake_cnn import dataset as dataset_module
from experiments.cifake_cnn.config import NEGATIVE_CLASS, POSITIVE_CLASS


class FakeDataset:
    """Small ImageFolder-like dataset for isolated unit tests."""

    def __init__(self, samples, class_to_idx=None):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.class_to_idx = class_to_idx or {"FAKE": 0, "REAL": 1}

    def __len__(self):
        return len(self.samples)


def make_fake_dataset(real_count=10, fake_count=10):
    samples = []
    for index in range(fake_count):
        samples.append((f"FAKE/fake_{index}.jpg", 1))
    for index in range(real_count):
        samples.append((f"REAL/real_{index}.jpg", 0))
    return FakeDataset(samples)


def test_class_counts_returns_correct_label_counts():
    dataset = make_fake_dataset(real_count=7, fake_count=5)

    assert dataset_module.class_counts(dataset) == {0: 7, 1: 5}


def test_train_val_split_is_disjoint_and_covers_dataset():
    dataset = make_fake_dataset(real_count=20, fake_count=20)

    train_indices, val_indices = dataset_module.make_train_val_split(
        dataset, seed=42, val_fraction=0.2
    )

    assert set(train_indices).isdisjoint(val_indices)
    assert set(train_indices) | set(val_indices) == set(range(len(dataset)))


def test_train_val_split_is_reproducible_with_same_seed():
    dataset = make_fake_dataset()

    first = dataset_module.make_train_val_split(dataset, seed=123)
    second = dataset_module.make_train_val_split(dataset, seed=123)

    assert first == second


def test_train_val_split_changes_with_different_seed():
    dataset = make_fake_dataset(real_count=50, fake_count=50)

    first = dataset_module.make_train_val_split(
        dataset, seed=42, val_fraction=0.2
    )
    second = dataset_module.make_train_val_split(
        dataset, seed=99, val_fraction=0.2
    )

    assert first[1] != second[1]


def test_train_val_split_preserves_class_proportions():
    dataset = make_fake_dataset(real_count=100, fake_count=100)

    train_indices, val_indices = dataset_module.make_train_val_split(
        dataset, seed=42, val_fraction=0.2
    )

    val_labels = [dataset.samples[index][1] for index in val_indices]
    train_labels = [dataset.samples[index][1] for index in train_indices]

    assert val_labels.count(0) == 20
    assert val_labels.count(1) == 20
    assert train_labels.count(0) == 80
    assert train_labels.count(1) == 80


def test_split_handles_unequal_class_sizes():
    dataset = make_fake_dataset(real_count=30, fake_count=10)

    train_indices, val_indices = dataset_module.make_train_val_split(
        dataset, seed=42, val_fraction=0.2
    )

    val_labels = [dataset.samples[index][1] for index in val_indices]

    assert val_labels.count(0) == 6
    assert val_labels.count(1) == 2
    assert len(train_indices) == 32


def test_verify_split_accepts_valid_split(capsys):
    dataset = make_fake_dataset()

    train_indices, val_indices = dataset_module.make_train_val_split(dataset)

    dataset_module.verify_split(dataset, train_indices, val_indices)

    assert "split OK" in capsys.readouterr().out


def test_verify_split_rejects_overlapping_indices():
    dataset = make_fake_dataset()

    with pytest.raises(AssertionError, match="overlap"):
        dataset_module.verify_split(dataset, [0, 1], [1, 2])


def test_verify_split_rejects_incomplete_coverage():
    dataset = make_fake_dataset()

    with pytest.raises(AssertionError):
        dataset_module.verify_split(dataset, [0, 1], [2, 3])


def test_verify_label_mapping_accepts_correct_labels(capsys):
    dataset = make_fake_dataset()

    dataset_module.verify_label_mapping(dataset, dataset)

    assert "label mapping OK" in capsys.readouterr().out


def test_verify_label_mapping_rejects_incorrect_label():
    dataset = FakeDataset([("FAKE/wrong.jpg", 0)])

    with pytest.raises(AssertionError, match="label mismatch"):
        dataset_module.verify_label_mapping(dataset, dataset)


def test_make_datasets_remaps_imagefolder_labels():
    fake_train = FakeDataset(
        [
            ("train/FAKE/fake.jpg", 0),
            ("train/REAL/real.jpg", 1),
        ]
    )
    fake_test = FakeDataset(
        [
            ("test/FAKE/fake.jpg", 0),
            ("test/REAL/real.jpg", 1),
        ]
    )

    with patch.object(
        dataset_module,
        "ImageFolder",
        side_effect=[fake_train, fake_test],
    ):
        train_ds, test_ds = dataset_module.make_datasets(image_size=32)

    assert train_ds.targets == [1, 0]
    assert test_ds.targets == [1, 0]
    assert train_ds.samples[0][1] == 1
    assert train_ds.samples[1][1] == 0
    assert POSITIVE_CLASS == "FAKE"
    assert NEGATIVE_CLASS == "REAL"


def test_transform_produces_expected_image_size():
    from PIL import Image

    transform = dataset_module._transform(32)
    image = Image.new("RGB", (100, 80))

    transformed = transform(image)

    assert tuple(transformed.shape) == (3, 32, 32)


def test_empty_split_is_rejected():
    dataset = make_fake_dataset()

    with pytest.raises(AssertionError, match="non-empty"):
        dataset_module.verify_split(dataset, [], list(range(len(dataset))))
