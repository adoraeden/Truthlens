# Project History — TruthLens to PRISM

## Origin: TruthLens

The project began as **TruthLens**, a multimodal deepfake and synthetic-media screening application developed for **NEURAL STORM 2026**, Problem Statement FT-02.

The original application explored how existing machine-learning models and forensic techniques could assist with the inspection of images, audio, and videos.

The project received **98th rank out of 329 participating teams**.

## Original application

The original application included:

- A React and TypeScript frontend.
- A Python backend built with FastAPI.
- Image screening using an externally developed pretrained Vision Transformer checkpoint.
- Audio screening using PellaV2 and its pretrained foundation model.
- Video analysis using sampled frames and, when available, an extracted audio track.
- Supporting forensic diagnostics such as attribution maps, noise residuals, Error Level Analysis, and image statistics.
- Initial evaluation scripts, reports, and automated tests.

The original image and audio model checkpoints were developed by external authors. Their use in the application is distinct from training a model from random initialisation.

## Transition toward PRISM

The project is evolving into **PRISM — Machine Learning & Synthetic Media Research**.

The original TruthLens application remains part of the repository as the project's application foundation and development history. The new direction places greater emphasis on understanding machine learning, developing independent models, conducting controlled experiments, and documenting findings.

## Existing independent CNN experiment

The repository includes `experiments/cifake_cnn/`, an isolated experiment that trains a custom CNN from random initialisation on CIFAKE.

The experiment includes:
- A custom CNN architecture.
- Dataset loading and a reproducible training/validation split.
- A training pipeline and checkpoint selection.
- Evaluation and analysis code.
- Unit tests and experiment documentation.

Its recorded training report gives 95.63% validation accuracy at the selected checkpoint. This is a result on the CIFAKE validation split; it does not establish performance on unfamiliar datasets or real-world media.

The held-out GenImage sample is intended for generalisation evaluation and should not be used to train or tune the model.

## Historical records

- [Earlier research log](../RESEARCH_LOG.md) — pre-hackathon research notes and early model checks.
- [CIFAKE CNN experiment](../experiments/cifake_cnn/README.md) — implementation, dataset split, architecture, and training pipeline.
- [CIFAKE CNN training results](../experiments/cifake_cnn/TRAINING_RESULTS.md) — recorded training run and observations.

## Attribution and transparency

The transition to PRISM does not retroactively change the origin of the earlier work. The application, externally pretrained models, datasets, and independent CNN experiment are described according to what they actually are. Future research contributions will be documented separately as they are completed.
