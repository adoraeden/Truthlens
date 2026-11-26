# PRISM — Machine Learning & Synthetic Media Research

**PRISM** is an evolving research project focused on understanding how machine learning models learn from visual information, discover patterns in images, and generalise to unfamiliar data.

The project began as **TruthLens**, a multimodal synthetic-media screening application developed for NEURAL STORM 2026. That application remains an important part of the project's history and technical foundation. PRISM builds on that experience by placing greater emphasis on learning, implementing, training, and evaluating machine-learning models ourselves.

> **From using existing AI models to understanding and developing machine-learning models.**

## Research direction

Our long-term interest is broader than a simple real-versus-AI classification task. We want to investigate how models learn visual information and what can be discovered from the patterns they learn.

Initial areas of interest include:

- Understanding neural networks and convolutional neural networks (CNNs) from the fundamentals.
- Learning how images are represented and how models recognise visual patterns.
- Extracting and comparing learned image representations.
- Exploring similarity, clustering, and anomaly detection.
- Training models from random initialisation and comparing different architectures.
- Testing models on unfamiliar datasets and images from different generators.
- Studying model behaviour, errors, limitations, and explainability.

The first research focus is **images**. Audio, video, and other directions may be explored later. These are research goals, not claims of completed capabilities or novel discoveries.

## Project development

PRISM is organised as a continuing research journey. The phases below describe the intended progression; their status will be updated as work is actually completed.

| Phase | Focus | Status |
|---|---|---|
| 0 | TruthLens application foundation: multimodal screening, pretrained model integration, forensic diagnostics, and initial evaluation | Historical foundation |
| 1 | Machine-learning foundations: datasets, neural networks, CNNs, training, and evaluation | Planned |
| 2 | Independent model development and controlled training experiments | Early work exists; further development planned |
| 3 | Visual representation research: image features, similarity, clustering, and unusual patterns | Planned |
| 4 | Dataset and architecture comparisons; controlled experiments | Planned |
| 5 | Generalisation, robustness, and explainability | Planned |
| 6 | Reproducible findings and a technical research report | Planned |

See the [PRISM Research Roadmap](docs/PRISM_ROADMAP.md) for more detail and the [Project History](docs/PROJECT_HISTORY.md) for the original TruthLens application.

## Existing research starting point

The repository already contains an independent CNN experiment:

- **Experiment:** `experiments/cifake_cnn/`
- **Dataset:** CIFAKE
- **Model:** A custom CNN with randomly initialised weights (not a pretrained model).
- **Training:** The existing experiment includes a training pipeline, evaluation code, tests, and recorded training results.
- **Reported result:** 95.63% validation accuracy at the selected checkpoint.

This is an initial result on the CIFAKE validation split, not evidence of general performance on unfamiliar datasets. Independent evaluation and further analysis are needed.

The experiment's README and [training results](experiments/cifake_cnn/TRAINING_RESULTS.md) document its setup and reported measurements. The held-out GenImage sample is intended for generalisation evaluation, not training.

## Two parts of the project's history

### 1. Original TruthLens application

The original application explored multimodal synthetic-media screening using externally developed pretrained models and supporting forensic analysis:

- Image analysis using a pretrained Vision Transformer.
- Audio analysis using PellaV2 and its pretrained foundation model.
- Video analysis using sampled frames and available audio.
- Forensic diagnostics, including attribution and image-analysis visualisations.
- A React frontend and Python/FastAPI backend.

These components are retained as the project's application foundation. They are not presented as models trained from scratch by this project.

### 2. PRISM research direction

The research direction focuses on understanding machine learning and developing our own experiments. The existing CIFAKE CNN is an early step in that direction. Future work will be documented with its research question, method, dataset, results, limitations, and reproducibility details.

## Repository structure

```text
.
├── frontend/                 # Original TruthLens web application
├── src/                      # Original application backend and detectors
├── experiments/
│   └── cifake_cnn/            # Independent CNN training experiment
├── data/                     # Existing datasets and samples
├── evaluation/               # Evaluation resources
├── reports/                  # Existing reports
├── docs/
│   ├── screenshots/          # Original application screenshots
│   ├── PRISM_ROADMAP.md      # Research phases and planned direction
│   └── PROJECT_HISTORY.md    # TruthLens application history
├── RESEARCH_LOG.md           # Earlier pre-hackathon research notes
└── README.md
```

## Technology

The existing application and experiments use technologies including Python, PyTorch, FastAPI, React, TypeScript, and Vite. Individual experiments may introduce additional tools when justified and documented.

## Research principles

- **Understand before implementing:** learn the concepts behind each model and experiment.
- **Be transparent:** distinguish pretrained components from models developed in this project.
- **Measure honestly:** report datasets, sample sizes, metrics, and limitations.
- **Keep experiments reproducible:** record configurations, seeds, data splits, and evaluation methods.
- **Learn from failures:** unsuccessful experiments and unexpected results are part of the research record.
- **Avoid unsupported claims:** an encouraging result on one dataset does not establish real-world reliability.

## Project origin

PRISM continues the work originally developed as TruthLens for **NEURAL STORM 2026**, Problem Statement FT-02: Deepfake & Synthetic Media Detection. The project received **98th rank out of 329 participating teams**.

The original demonstration and application screenshots are retained as part of the project's history. The earlier development and evaluation notes are available in [RESEARCH_LOG.md](RESEARCH_LOG.md).

---

**PRISM — Learning how machines understand images.**
