# PRISM Research

This folder contains our machine-learning experiments.

## Current experiment: CIFAKE CNN

We built and trained our own CNN to classify images as real or AI-generated.

**Current results:**
- CIFAKE test accuracy: **95.67%**
- GenImage exploratory sample accuracy: **49.25%**

The difference between these results shows that the model struggles with images from other sources. Our current focus is improving that generalisation.

## Roadmap

- [x] **Phase 1 — Build the CNN:** model, dataset loading, labels and data split.
- [x] **Phase 2 — Train and test:** training pipeline and official CIFAKE evaluation.
- [ ] **Phase 3 — Improve generalisation:** investigate the failure and test better training data and augmentations.
- [ ] **Phase 4 — Test image robustness:** compression, resizing, blur, noise and screenshots.
- [ ] **Phase 5 — Explain the model:** visualise which image regions influence its predictions.
- [ ] **Phase 6 — Explore advanced models:** architecture comparisons and feature combinations.

For the detailed checklist and results, see the [CIFAKE CNN experiment README](experiments/cifake_cnn/README.md).

## Dataset

The CIFAKE dataset is not stored in Git because of its size. Download it separately and place it in `research/data/kaggle_image/`, with `train/` and `test/` folders.

## Research principles

- Keep training, validation and final testing separate.
- Record the datasets and settings used for each experiment.
- Report mistakes and limitations as well as good results.
- Keep research models separate from the production TruthLens detector.
