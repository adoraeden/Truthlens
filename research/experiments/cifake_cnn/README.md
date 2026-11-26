# PRISM — CIFAKE CNN Experiment (from-scratch training)

**Status:** implementation and verification complete. Phase 2 (training pipeline in `training.py`) is implemented; see the Training section below.

## Purpose

A first independent PRISM machine-learning experiment: train a custom CNN
**from random initialisation** on the CIFAKE dataset to classify AI-generated
(`FAKE`) vs. genuine (`REAL`) images. The goal is to understand CNN training
end-to-end — data → model → loss → gradients → checkpoints — before building
the broader PRISM research programme.

## Why a CNN from scratch?

- The original TruthLens application uses a pretrained ViT image detector (see `TruthLens/src/image_detector.py`).
  This experiment is intentionally independent: our own architecture, our own
  training, from random init.
- CIFAKE is a balanced, local benchmark: 50k real / 50k fake in `train/`,
  10k / 10k in `test/` (see `research/data/kaggle_image/`).

## Dataset and label mapping

- Root: `research/data/kaggle_image/` (download locally; dataset files are git-ignored).
- **Mapping (explicit, not the default): `FAKE = 1` (positive class), `REAL = 0`.**
  `dataset.py` remaps `ImageFolder`'s alphabetical assignment so the positive
  class is always 1 (`verify_label_mapping` checks every file).
- `research/data/kaggle_image/train/` is the training + validation directory;
  `research/data/kaggle_image/test/` is the official test set (never used for
  training/validation).

## Train / validation / test separation

- `make_train_val_split()` produces a **reproducible 90/10 split of the CIFAKE
  train directory** (`seed=42`): ~90,000 train / 10,000 val.
- Train and val indices are guaranteed disjoint and cover the directory
  (`verify_split`).

## Resolution: 32 (smoke) → 64 (main)

- CIFAKE images are uniformly **32×32** RGB JPEGs (verified in Stage 0).
- We start the smoke test at **32×32** (`IMAGE_SIZE_SMOKE`) — the native size,
  fastest to iterate on the T4.
- The main run uses **64×64** (`IMAGE_SIZE_MAIN`) — 2× the pixels, better
  feature granularity, still cheap on a T4 for this small CNN.
  (128×128 is available later if compute allows.)

## Architecture (actual, random init)

```
Input   : [B, 3, H, W]  (32 or 64, RGB)
Stages  : 5 convolutional stages with increasing widths: 3 -> 32 -> 64 -> 96 -> 128 -> 160
          Each stage:
            Conv2d(in, w, k=3, s=2, p=1) -> BatchNorm2d(w) -> ReLU
            Conv2d(w, w, k=3, s=1, p=1)  -> BatchNorm2d(w) -> ReLU
Head    : AdaptiveAvgPool2d(1) -> Flatten -> Linear(160, 64) -> ReLU -> Linear(64, 1)
Output  : [B, 1] raw logits (no sigmoid inside the model)
```

**Actual trainable parameters: 889,633.** Verified by running `run_shape_test()`
(`python -B -m research.experiments.cifake_cnn.model` from the repository root):

```
input 32x32: logits shape (4, 1); parameters: 889,633
input 64x64: logits shape (4, 1); parameters: 889,633
```

Key components:
- `Conv2d`: learns filters (features from pixels).
- `BatchNorm2d`: normalizes activations, helps training stability.
- `ReLU`: introduces non-linearity.
- `AdaptiveAvgPool2d(1)`: global average pooling — collapses the spatial map to
  one value per channel (robust, resolution-independent).
- Small `Linear` head: maps pooled features to the single classification logit.

## Training

Train the custom CNN from scratch on CIFAKE (run from the repo root):

```powershell
.venv\Scripts\python.exe -B -m experiments.cifake_cnn.training
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--image-size` | `64` | Square training input size |
| `--batch-size` | `64` | Training batch size |
| `--epochs` | `20` | Maximum epochs |
| `--device` | auto | `cuda` or `cpu` |
| `--seed` | `42` | Random seed |
| `--output-dir` | `research/experiments/cifake_cnn/outputs` | Output directory |

How it works (`training.py`):

- Uses the existing `CifakeCNN` and the existing reproducible train/val split
  (`make_train_val_split`) — no duplicated logic.
- Loss: `BCEWithLogitsLoss` (the model returns raw logits).
- Optimizer: `AdamW` with the configured `LEARNING_RATE` and `WEIGHT_DECAY`.
- Tracks train + val loss and accuracy every epoch.
- Saves the best-epoch checkpoint by val loss (model state, optimizer state,
  epoch, `val_loss`/`val_accuracy`, `best_val_loss`).
- Early stopping using `EARLY_STOPPING_PATIENCE` from `config.py`.

Outputs (isolated under `research/experiments/cifake_cnn/outputs/` by default):

- `checkpoints/best_cifake_cnn.pt` — best-epoch checkpoint.
- `history.csv` — per-epoch rows (append-only; earlier epochs are preserved).

Unit tests (small synthetic models/datasets only — no full CIFAKE training):

```powershell
.venv\Scripts\python.exe -B -m pytest experiments\cifake_cnn\test_training.py -q
```

Note: the official CIFAKE `test/` directory is never used for training or validation.

## GenImage — held out

`data/genimage_validation_sample/` (8 generator dirs: ADM, BigGAN, glide,
Midjourney, stable_diffusion_v_1_4, stable_diffusion_v_1_5, VQDM, wukong;
manifest: 200 `ai` + 200 `nature`) is **separate from this experiment** and
**must remain held out for generalisation evaluation only** (Stage 4).

## Relationship to the original TruthLens application

Nothing under `src/` is modified; this package is additive and independent of
the production image detector.

## Files

- `config.py` — hyperparameters (batch 64, lr 3e-4, wd 0.01, 20 epochs, seed 42, ...).
- `model.py` — `CifakeCNN`, `count_parameters`, `run_shape_test`.
- `dataset.py` — CIFAKE loading, label remapping, 90/10 split, verification helpers.
- `training.py` — training pipeline: loaders, train/eval epochs, AdamW + BCEWithLogitsLoss,
  early stopping, best-checkpoint saving, history CSV, `main()` entry point.
- `test_training.py` — unit tests for the training utilities (small synthetic models/datasets).
- `__init__.py` — package marker.

## Verification results (from implementation time)

- Shape test at 32×32: output `(4, 1)` ✅
- Shape test at 64×64: output `(4, 1)` ✅
- Parameter count: **889,633** ✅
- Label mapping: `FAKE=1`, `REAL=0` — verified on all 30,000 train/test files ✅
- Split: 90,000 train / 10,000 val with seed 42, disjoint and reproducible ✅
  (first val indices: `3813, 99183, 80737, 65674, 53670`)
