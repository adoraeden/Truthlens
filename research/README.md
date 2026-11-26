# PRISM Research

This directory contains independent machine-learning experiments and their research-specific resources.

## Current experiment

- [CIFAKE CNN](experiments/cifake_cnn/): a convolutional neural network trained from random initialisation to study image features and cross-generator generalisation.
- [Dataset metadata](dataset-metadata.json): source metadata for CIFAKE.
- [Research environment requirements](requirements-research.txt): pinned dependencies for the research environment.

The CIFAKE dataset is not committed to Git because of its size. Place the downloaded dataset at `research/data/kaggle_image/`, with `train/` and `test/` subdirectories. The experiment's dataset loader expects that location.

## Research principle

Keep training, validation, and final evaluation distinct. Treat results as evidence about the tested dataset and conditions, not as a guarantee of real-world performance.
