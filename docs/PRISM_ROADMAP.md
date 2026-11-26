# PRISM Research Roadmap

This document describes the intended research progression for PRISM. It is a living plan: phases move to in-progress or completed only when the corresponding work has actually been carried out and documented.

## Overall direction

PRISM investigates how machine-learning models learn from visual information, discover patterns in images, and generalise to unfamiliar data.

The project begins with image research. The aim is not to restrict every experiment to a real-versus-AI label. We will first understand the fundamentals, then investigate learned image representations, similarity, clustering, unusual patterns, and model behaviour.

## Phase 0 — TruthLens application foundation

**Status: Historical foundation**

The project began as TruthLens, a multimodal synthetic-media screening application developed for NEURAL STORM 2026.

This phase includes the original application, integration of externally pretrained image and audio models, video analysis workflow, forensic diagnostics, and initial evaluation. See [PROJECT_HISTORY.md](PROJECT_HISTORY.md).

## Phase 1 — Machine-learning foundations

**Status: Planned**

Build a clear understanding of the ideas needed for independent research.

Topics:
- What artificial intelligence and machine learning mean.
- How a computer represents an image as numbers.
- What a neural network is and how it learns from examples.
- What a CNN does and why it is useful for images.
- Training, validation, testing, loss, and evaluation.
- How to read and understand the existing CIFAKE CNN code.

**Expected outcome:** Be able to explain the basic operation of the existing model and its training process in plain language.

## Phase 2 — Independent model development

**Status: Early work exists; further development planned**

Build on the existing `experiments/cifake_cnn/` experiment.

Topics:
- Understand the current CNN architecture and training pipeline.
- Reproduce its documented training process where practical.
- Evaluate the saved model on data that was not used for training or model selection.
- Learn how changes to architecture and training settings affect results.
- Keep each experiment isolated and record its configuration.

**Expected outcome:** A well-understood baseline and a reproducible record of independent model experiments.

## Phase 3 — Visual representation research

**Status: Planned**

Investigate what information a model learns about images, beyond its final classification.

Possible investigations:
- Extract learned representations from different parts of a CNN.
- Compare representations from different images.
- Visualise high-dimensional representations in two dimensions.
- Explore whether similar images form groups.
- Investigate whether different image sources or generators produce distinguishable patterns.
- Explore anomaly detection and region-level analysis where appropriate.

**Expected outcome:** Experiments and visualisations that help us understand what the model has learned. We will not assume that learned patterns are reliable indicators of image origin without testing them.

## Phase 4 — Controlled experiments and comparisons

**Status: Planned**

Change one factor at a time and study its effect.

Possible factors:
- Image resolution and preprocessing.
- Training data and data balance.
- Data augmentation.
- Model depth and width.
- Different CNN architectures.
- Training settings and regularisation.

**Expected outcome:** Fair comparisons supported by recorded methods and results, including unsuccessful experiments.

## Phase 5 — Generalisation, robustness, and explainability

**Status: Planned**

Investigate how models behave beyond the data they were trained on.

Possible investigations:
- Test images from generators not represented in training data.
- Study the effect of compression, resizing, and other image changes.
- Examine false positives and false negatives.
- Compare performance across image sources and categories.
- Visualise which image regions influence model outputs.
- Identify limitations and possible dataset biases.

**Expected outcome:** A more careful understanding of where a model works, where it fails, and why.

## Phase 6 — Research findings and reporting

**Status: Planned**

Bring the experiments together into a reproducible technical report.

The report should include:
- Research questions and hypotheses.
- Datasets, sources, licences, and split strategy.
- Model descriptions and experiment settings.
- Evaluation metrics and visualisations.
- Results, comparisons, and error analysis.
- Limitations, failed experiments, and future questions.

A paper or publication may be considered if the work eventually supports a meaningful contribution. Publication or novelty is not assumed in advance.

## Future exploration

After the initial image-focused work, PRISM may explore:
- Audio representation and synthetic speech analysis.
- Video and temporal patterns.
- Multimodal learning.
- Other machine-learning topics, including artificial-life or ecosystem simulations, if a clear research question emerges.

These are future possibilities, not current deliverables.

## How we record an experiment

Each substantial experiment should record:

1. **Question:** What are we trying to find out?
2. **Hypothesis:** What do we expect, and why?
3. **Method:** What data and model will we use?
4. **Controls:** What will remain unchanged?
5. **Results:** What happened?
6. **Interpretation:** What might explain the result?
7. **Limitations:** What can we not conclude?
8. **Reproduction:** How can the experiment be run again?

PRISM values understanding and honest evidence over adding features or chasing a high accuracy number.
