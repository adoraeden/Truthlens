# PRISM — CIFAKE CNN Experiment (from-scratch training)

**Current status:** Phases 1–2 are complete. Phase 3 (generalisation) is the active research phase. The roadmap below distinguishes completed work from planned experiments.

## Purpose

Train and study a custom CNN **from random initialisation** on the CIFAKE dataset to classify AI-generated (`FAKE`) vs. genuine (`REAL`) images. The goal is to understand CNN training end-to-end — data → model → loss → gradients → checkpoints — and, most importantly, investigate whether learned image features generalise beyond the dataset used for training.

## Why a CNN from scratch?

- The original TruthLens application uses a pretrained ViT image detector (see `TruthLens/src/image_detector.py`). This experiment is intentionally independent: it uses our own architecture and random initialisation.
- CIFAKE is a balanced benchmark: 50,000 real and 50,000 fake images in `train/`, and 10,000 of each class in `test/` (see `research/data/kaggle_image/`).
- The experiment is additive and does not modify the production TruthLens detector.

## Dataset and label mapping

- Root: `research/data/kaggle_image/` (download locally; dataset files are git-ignored).
- **Mapping: `FAKE = 1` (positive class), `REAL = 0`.** `dataset.py` remaps ImageFolder's alphabetical assignment so this convention is explicit.
- `research/data/kaggle_image/train/` is used for training and validation.
- `research/data/kaggle_image/test/` is the official test set and must remain separate from training and model-selection decisions.

## Baseline architecture

```
Input   : [B, 3, H, W] (32 or 64, RGB)
Stages  : 5 convolutional stages, widths 3 -> 32 -> 64 -> 96 -> 128 -> 160
          Each stage:
            Conv2d(in, w, k=3, s=2, p=1) -> BatchNorm2d(w) -> ReLU
            Conv2d(w, w, k=3, s=1, p=1)  -> BatchNorm2d(w) -> ReLU
Head    : AdaptiveAvgPool2d(1) -> Flatten -> Linear(160, 64) -> ReLU -> Linear(64, 1)
Output  : [B, 1] raw logits (no sigmoid inside the model)
```

**Trainable parameters: 889,633.** The architecture was shape-tested at 32×32 and 64×64.

## Completed work

### Phase 1 — Foundation and data pipeline

**Status: Complete**

- Implemented the custom CNN architecture from random initialisation.
- Implemented CIFAKE loading and explicit `FAKE=1` / `REAL=0` label mapping.
- Implemented reproducible, stratified 90/10 training/validation splitting with seed 42.
- Verified that training and validation indices are disjoint and cover the training directory.
- Added focused tests for the model and dataset utilities.

### Phase 2 — Training pipeline and CIFAKE baseline

**Status: Complete**

- Implemented training and validation loops using `BCEWithLogitsLoss` and AdamW.
- Added deterministic seeding, early stopping, best-validation-loss checkpointing and append-only CSV training history.
- Trained on a Google Colab NVIDIA Tesla T4 at 64×64 resolution, batch size 64, with a maximum of 20 epochs and early-stopping patience of 5.
- Training stopped after 12 epochs; the best checkpoint was selected at epoch 6.

**Recorded CIFAKE results**

| Evaluation | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Best validation checkpoint | 95.63% | — | — | — |
| Official CIFAKE test (20,000 images) | 95.67% | 95.89% | 95.43% | 95.66% |

Official-test confusion matrix (rows = actual REAL/FAKE, columns = predicted REAL/FAKE):

```
                 Predicted
                 REAL   FAKE
Actual REAL      9591    409
Actual FAKE       457   9543
```

These are benchmark results on CIFAKE. They do not establish performance on unfamiliar generators or real-world media.

## Initial cross-dataset diagnostic

An exploratory evaluation was previously run on a **400-image GenImage sample** containing 200 AI-generated and 200 real/nature images drawn across eight generator folders.

| Metric | Recorded result |
|---|---:|
| Accuracy | 49.25% |
| Precision (FAKE positive) | 49.6% |
| Recall (FAKE positive) | 92.5% |
| F1 (FAKE positive) | 64.57% |

Confusion matrix (rows = actual REAL/FAKE, columns = predicted REAL/FAKE):

```
                 Predicted
                 REAL   FAKE
Actual REAL        12    188
Actual FAKE        15    185
```

The model predicted FAKE for 373 of the 400 images and correctly classified only 12 of the 200 real images. This indicates a severe false-positive problem on this small sample. The result is a diagnostic warning, not a reliable estimate of general real-world performance.

**Research consequence:** CIFAKE performance is strong, but cross-dataset generalisation is not. We need to investigate dataset and preprocessing differences, model behaviour and source diversity rather than simply chase a higher CIFAKE accuracy.

## Research roadmap

### Phase 3 — Generalisation and cross-generator performance

**Status: Active — current phase**

**Objective:** understand and improve performance on data that differs from CIFAKE, while keeping comparisons fair and avoiding evaluation leakage.

1. **3.1 — Data and evaluation audit**
   - Verify image dimensions, formats, colour channels, class labels and preprocessing across datasets.
   - Compare content and source distributions for real and generated images.
   - Inspect prediction scores, confusion matrices and false-positive/false-negative examples by generator.
   - Record exactly which preprocessing is applied in training and evaluation.
   - Create a separate development set for experiment selection. The previously inspected 400-image GenImage sample is a reference diagnostic and must not be repeatedly used to tune the model.

2. **3.2 — Controlled augmentation experiments**
   - Establish an unchanged baseline run for comparison.
   - Test spatial augmentation (for example, random resized crops and horizontal flips).
   - Test image-quality transformations (for example, JPEG compression, resizing, mild blur and controlled noise).
   - Test moderate colour and appearance variation.
   - Run ablations one group at a time before testing a combined policy.
   - Apply stochastic augmentation only to training data; keep validation and evaluation preprocessing deterministic.

3. **3.3 — Training-source diversity**
   - Investigate adding suitable images from multiple generator families.
   - Keep real and generated examples reasonably comparable in content and processing.
   - Separate training, development and final evaluation by source or generator where possible.
   - Record dataset provenance, licences, class balance and generator coverage.

4. **3.4 — Independent generalisation evaluation**
   - Evaluate the selected model on generator families not used for training.
   - Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
   - Break down results by generator and class, not just as one aggregate score.
   - Compare against the original CIFAKE baseline and report limitations.
   - Use multiple random seeds for promising configurations where compute permits.

**Phase 3 success criterion:** demonstrate a repeatable improvement in external-data performance without hiding class-specific failures or materially sacrificing the CIFAKE benchmark. If an experiment does not improve generalisation, document that result rather than presenting it as a success.

### Phase 4 — Real-world image robustness

**Status: Planned**

- Evaluate controlled JPEG compression levels, resizing, blur, noise, crops and screenshot-like transformations.
- Measure performance degradation as transformation severity increases.
- Distinguish robustness to image processing from generalisation to new generators.
- Keep transformed evaluation copies separate from training unless a specific robustness-training experiment is being tested.

### Phase 5 — Explainability and feature investigation

**Status: Planned**

- Explore Grad-CAM or related attribution methods to visualise regions influencing predictions.
- Inspect feature maps and compare correct predictions with false positives and false negatives.
- Investigate spatial and frequency-domain signals as separate evidence sources.
- Check whether explanations are stable and meaningful; visual heatmaps are diagnostic aids, not proof of manipulation.

### Phase 6 — Advanced model research

**Status: Planned**

- Compare carefully selected CNN architecture variants.
- Investigate spatial/frequency feature fusion and alternative classification heads.
- Study calibration and uncertainty so that weak or unfamiliar inputs are not presented with unjustified confidence.
- Consider ensembles only if independent evaluation shows a meaningful benefit.
- Keep experimental models isolated from the production TruthLens detector until they have passed appropriate evaluation.

## Research and reproducibility rules

- Do not train on the official CIFAKE test set.
- Do not repeatedly tune against the previously inspected 400-image GenImage diagnostic sample; use a separate development set and preserve an independent final evaluation set.
- Change one major factor at a time during ablation experiments.
- Record configuration, seed, dataset version, preprocessing, checkpoint and metrics for every run.
- Compare models on the same evaluation examples and report per-class results.
- Keep model checkpoints, datasets and large generated artifacts out of Git unless there is a specific, appropriate reason to version them.
- Treat detector outputs as experimental signals, not proof that an image is fake or authentic.

## Training and verification commands

Run from the repository root:

The current runner reproduces the existing training recipe. A unique output directory keeps earlier checkpoints and histories preserved:

```powershell
.venv\Scripts\python.exe -B -m research.experiments.cifake_cnn.training --output-dir research/experiments/cifake_cnn/outputs/phase3_baseline_reproduction
```

Run focused experiment tests:

```powershell
.venv\Scripts\python.exe -B -m pytest research\experiments\cifake_cnn -q
```

Evaluate the selected checkpoint on the official CIFAKE test split only when the experiment is ready for a final benchmark comparison. Adjust the two paths to match the run being evaluated:

```powershell
.venv\Scripts\python.exe -B -m research.experiments.cifake_cnn.evaluate --checkpoint research/experiments/cifake_cnn/outputs/phase3_baseline_reproduction/checkpoints/best_cifake_cnn.pt --output research/experiments/cifake_cnn/outputs/phase3_baseline_reproduction/official_test_results.json
```

The local research environment may be CPU-only. The full CIFAKE training run used a Colab T4 GPU. Select a GPU runtime for substantial training runs.

## Experiment files

- `config.py` — image sizes and training hyperparameters.
- `model.py` — `CifakeCNN`, parameter counting and shape test.
- `dataset.py` — CIFAKE loading, label remapping and reproducible split.
- `training.py` — training/validation loops, checkpointing, early stopping and history.
- `evaluate.py` — official CIFAKE test evaluation and JSON report.
- `analysis.py` — training-history analysis.
- `test_*.py` — focused tests using synthetic data; they do not run full CIFAKE training.

## References and provenance

CIFAKE contains REAL images derived from CIFAR-10 and FAKE images generated using Stable Diffusion 1.4. See `research/dataset-metadata.json` for recorded dataset metadata and the dataset publication details. Preserve the dataset's required attribution when publishing results.
