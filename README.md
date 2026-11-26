# PRISM
### Machine Learning & Synthetic Media Research

PRISM is the research direction of this repository: a practical exploration of machine learning, visual representations, synthetic-media detection, and how well learned patterns generalise beyond the data used to train a model.

This repository grew out of **TruthLens**, the original multimodal synthetic-media screening application. TruthLens remains part of the project as its application and baseline system; PRISM builds on that foundation through independent experiments and evaluation.

## Explore the project

| Area | What you'll find |
|---|---|
| [Research](research/) | PRISM experiments, model development, datasets and research notes |
| [Original application — TruthLens](TruthLens/) | The original frontend, backend, pretrained detector integrations, tests, and historical evaluation work |

## Current research direction

The first independent experiment is a convolutional neural network (CNN) trained from random initialisation on CIFAKE.

**Research question:** Does a CNN trained on CIFAKE learn indicators that generalise to images from unseen generators, or does it rely on dataset-specific patterns?

The experiment is kept separate from TruthLens' pretrained image detector. Its training, validation, and held-out evaluation are documented in [the CIFAKE CNN experiment](research/experiments/cifake_cnn/).

## Project progression

1. **TruthLens:** integrated pretrained image and audio models into a multimodal application, with a video analysis pipeline and supporting forensic diagnostics.
2. **PRISM:** use that application as a baseline, then investigate model behaviour through independent training, controlled evaluation, and generalisation experiments.

Results are reported with their dataset and evaluation limitations. Performance on one benchmark is not presented as proof of reliable real-world synthetic-media detection.

## Repository layout

```text
.
├── README.md
├── research/
│   ├── experiments/
│   │   └── cifake_cnn/
│   ├── dataset-metadata.json
│   └── requirements-research.txt
├── TruthLens/
│   ├── frontend/
│   ├── src/
│   ├── tests/
│   ├── evaluation/
│   ├── reports/
│   └── ...
└── .github/
```

## Getting started

- To explore the research experiment, start with [research/README.md](research/README.md) and the [CIFAKE CNN experiment notes](research/experiments/cifake_cnn/README.md).
- To explore the original application, start with [TruthLens/README.md](TruthLens/README.md).

PRISM is an evolving research project. New experiments and conclusions will be added as they are implemented and evaluated.
