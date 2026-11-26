# PRISM — CIFAKE CNN Experiment

## What is this?

This project explores whether a custom CNN (convolutional neural network) can distinguish AI-generated images from real images.

We built the CNN ourselves rather than starting with a pretrained model. The experiment is separate from the production TruthLens detector.

Our main goal is not just to get a high score on one dataset. We want to understand whether the model can also work on images from generators and sources it has not seen before.

## Current results

We trained the CNN on CIFAKE using a Google Colab NVIDIA T4 GPU.

| Result | Score |
|---|---:|
| CIFAKE validation accuracy | **95.63%** |
| CIFAKE official test accuracy | **95.67%** |
| CIFAKE test F1 score | **95.66%** |

The official CIFAKE test contains 20,000 images. These results are good for this dataset, but they do not prove that the model works equally well on images from other sources.

### Our generalisation problem

We also tried the model on a small, separate sample of 400 GenImage pictures: 200 AI-generated and 200 real.

| Result | Score |
|---|---:|
| Accuracy | **49.25%** |
| Real images correctly identified | **12 out of 200** |
| AI-generated images correctly identified | **185 out of 200** |

The model labelled most of the real images as fake. This is a serious false-positive problem and shows that strong CIFAKE results do not automatically transfer to other datasets.

This 400-image sample is now a **diagnostic reference**. We should not keep adjusting the model against the same sample. We need separate development data and an independent final evaluation.

---

# Project Roadmap

Use the checkboxes to track our progress. Completed work is checked; unchecked items are still to do.

## Phase 1 — Build the CNN

- [x] Design and implement our own CNN architecture.
- [x] Set up the CIFAKE dataset loader.
- [x] Set the labels consistently: REAL = 0 and FAKE = 1.
- [x] Create a repeatable training and validation split.
- [x] Add tests for the model and dataset.

**Status: Complete**

## Phase 2 — Train and test the first model

- [x] Build the training and validation process.
- [x] Add loss tracking, checkpoint saving and early stopping.
- [x] Train the CNN on CIFAKE using a GPU.
- [x] Select the best model using validation results.
- [x] Evaluate it on the separate official CIFAKE test set.
- [x] Record the training and evaluation results.
- [x] Run the focused experiment tests (43 passed).

**Status: Complete**

## Phase 3 — Improve performance on unfamiliar images

**This is our current phase.** We want the CNN to work better on images from sources and AI generators that were not part of its original training.

### 3.1 Understand the problem

- [ ] Compare image sizes, formats and colour information across datasets.
- [ ] Check that labels and image preprocessing are consistent.
- [ ] Study the model's predictions and confidence scores.
- [ ] Inspect examples that were incorrectly labelled as fake or real.
- [ ] Compare results across different AI generators.
- [ ] Prepare separate development data for choosing improvements.

### 3.2 Try better training techniques

- [ ] Create a fresh run of the original model as a comparison baseline.
- [ ] Try random crops and horizontal flips.
- [ ] Try resizing and JPEG compression.
- [ ] Try mild blur and image noise.
- [ ] Try small brightness, contrast and colour changes.
- [ ] Test each change separately before combining useful ones.

### 3.3 Use more varied training images

- [ ] Find suitable real and AI-generated images from additional sources.
- [ ] Include images from different AI generator families.
- [ ] Keep the real and generated image collections reasonably comparable.
- [ ] Record the source, licence and number of images used.
- [ ] Keep final evaluation sources separate from training data.

### 3.4 Check whether the improvements really work

- [ ] Evaluate on sources not used for training or model selection.
- [ ] Compare the new model with the original CIFAKE baseline.
- [ ] Report accuracy, precision, recall, F1 and balanced accuracy.
- [ ] Check false positives and false negatives for each source.
- [ ] Repeat promising experiments with more than one random seed.
- [ ] Document improvements, regressions and limitations.

**Phase 3 goal:** improve performance on unfamiliar images without hiding mistakes or relying on CIFAKE accuracy alone.

## Phase 4 — Test real-world image quality changes

- [ ] Test JPEG compression at different quality levels.
- [ ] Test resized and blurry images.
- [ ] Test noisy images and screenshots.
- [ ] Test cropped images.
- [ ] Measure how each change affects the results.

## Phase 5 — Help us understand the model's decisions

- [ ] Add Grad-CAM or a similar visualisation to highlight influential image regions.
- [ ] Inspect what the model focuses on when it is correct.
- [ ] Inspect what it focuses on when it makes mistakes.
- [ ] Explore image-frequency patterns.
- [ ] Check whether visual explanations are stable and useful.

## Phase 6 — Explore more advanced models

- [ ] Compare other CNN architectures.
- [ ] Experiment with combining image and frequency features.
- [ ] Explore confidence calibration and uncertainty.
- [ ] Test whether an ensemble of models provides a real improvement.
- [ ] Only consider connecting a research model to TruthLens after proper evaluation.

---

## Important experiment rules

- Never train on the official CIFAKE test set.
- Keep training, validation and final evaluation data separate.
- Do not repeatedly tune against the previously inspected 400-image GenImage sample.
- Change one major training setting at a time so we can understand what helped.
- Save each experiment's settings, results and model checkpoint separately.
- Keep this research experiment separate from TruthLens's production detector.
- A model prediction is an indication, not proof that an image is real or fake.

## Project files

- `model.py` — our CNN architecture.
- `dataset.py` — dataset loading and splitting.
- `training.py` — model training.
- `evaluate.py` — evaluation on the official CIFAKE test set.
- `analysis.py` — training-history analysis.
- `test_*.py` — automated tests.

## Running the experiment

Run commands from the repository root.

Run the focused tests:

```powershell
.venv\Scripts\python.exe -B -m pytest research\experiments\cifake_cnn -q
```

The existing training script can reproduce the original training setup. Give every new run its own output folder so previous results are preserved:

```powershell
.venv\Scripts\python.exe -B -m research.experiments.cifake_cnn.training --output-dir research/experiments/cifake_cnn/outputs/phase3_baseline_reproduction
```

The local PyTorch installation may be CPU-only. The original full training was performed on a Google Colab T4 GPU, which is the recommended option for substantial training runs.

## Dataset information

CIFAKE contains real images derived from CIFAR-10 and AI-generated images produced using Stable Diffusion 1.4. See [dataset-metadata.json](../../dataset-metadata.json) for the recorded dataset information and attribution details.
