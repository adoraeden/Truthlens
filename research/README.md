# PRISM Research

This directory contains independent machine-learning experiments and their research-specific resources.

## Current experiment

- [CIFAKE CNN](experiments/cifake_cnn/): a convolutional neural network trained from random initialisation to study image features and cross-generator generalisation.
- [CIFAKE CNN research roadmap](experiments/cifake_cnn/README.md#research-roadmap): completed phases, baseline results, the current generalisation work and planned research.
- [Dataset metadata](dataset-metadata.json): source metadata for CIFAKE.
- [Research environment requirements](requirements-research.txt): pinned dependencies for the research environment.

The CIFAKE dataset is not committed to Git because of its size. Place the downloaded dataset at `research/data/kaggle_image/`, with `train/` and `test/` subdirectories. The experiment's dataset loader expects that location.

## Research progress

| Phase | Focus | Status |
|---|---|---|
| 1 | Custom CNN foundation, dataset loader, label mapping and reproducible split | Complete |
| 2 | Training pipeline, checkpointing, validation and official CIFAKE benchmark | Complete |
| 3 | Cross-dataset and cross-generator generalisation | Active |
| 4 | Robustness to compression, resizing, blur, noise and other image transformations | Planned |
| 5 | Explainability and feature investigation | Planned |
| 6 | Advanced architectures, feature fusion and calibration research | Planned |

The initial CIFAKE baseline reached 95.67% accuracy on the official CIFAKE test split. A separate, small 400-image GenImage diagnostic sample produced 49.25% accuracy and exposed a severe false-positive problem. The sample is a diagnostic reference, not a reliable estimate of real-world performance; the detailed metrics and limitations are documented in the experiment README.

## Research principles

- Keep training, validation, development and final evaluation distinct.
- Do not use the official CIFAKE test split for training or model selection.
- Do not repeatedly tune against the previously inspected GenImage diagnostic sample.
- Record dataset provenance, preprocessing, seeds, configurations and per-class metrics.
- Treat results as evidence about the datasets and conditions tested, not as a guarantee of real-world performance.
- Keep experimental CNN work independent of the production TruthLens detector until an experiment has been properly evaluated.
