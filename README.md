# PRISM — CIFAKE CNN Research Experiment

## 1. Project Overview

PRISM is a research project exploring whether a custom Convolutional Neural Network (CNN) can distinguish AI-generated images from real images.

The CNN is implemented from scratch rather than relying on a pretrained image-classification model.

This research investigates more than classification accuracy on a single dataset. Its primary objective is to understand how well a model trained on known image sources performs when evaluated on images from different datasets and AI generator families.

**Research priorities:**
- Build and evaluate a custom CNN architecture.
- Establish a reliable CIFAKE baseline.
- Investigate the model's generalisation limitations.
- Experiment with image augmentation and more diverse training data.
- Evaluate performance across different image sources and generators.
- Study false positives, false negatives and real-world image transformations.

This research project is separate from the production TruthLens detector. Research results must be independently evaluated before any consideration of production integration.

---

## 2. Model A — CIFAKE Baseline

Model A is the original custom CNN trained using the CIFAKE dataset.

It establishes the baseline against which subsequent experiments are investigated.

### CIFAKE results

| Metric | Model A |
|---|---:|
| Validation accuracy | 95.63% |
| Official test accuracy | **95.67%** |
| Test F1-score | 95.66% |
| Official test images | 20,000 |

The model achieved strong performance on the official CIFAKE test set.

However, high accuracy on CIFAKE does not necessarily indicate that the model can distinguish real and generated images from other sources.

### Initial GenImage evaluation

The baseline was also evaluated on a separate sample of 400 GenImage images.

| Metric | Model A |
|---|---:|
| Accuracy | 49.25% |
| AI images correctly identified | 185 / 200 |
| Real images correctly identified | 12 / 200 |
| Precision | 49.60% |
| Recall | 92.50% |
| F1-score | 64.57% |

The model correctly identified most generated images but incorrectly classified a large proportion of real images as AI-generated.

This revealed a significant generalisation problem and motivated the subsequent experiments.

The original 400-image sample is treated as a diagnostic reference rather than a dataset for repeated model tuning.

---

## 3. Model B — JPEG Compression Augmentation

Model B investigates whether image-quality augmentation can improve the baseline's ability to recognise generated images from different sources.

The CNN architecture remains the same, while random JPEG compression augmentation is introduced during training.

### CIFAKE results

| Metric | Model B |
|---|---:|
| Accuracy | 95.605% |
| Precision | 96.042% |
| Recall | 95.130% |
| F1-score | 95.584% |

### GenImage results

A historical 400-image GenImage evaluation produced the following results:

| Metric | Model B |
|---|---:|
| Accuracy | 52.75% |
| AI recall | 84.50% |
| Real specificity | 21.00% |
| Precision | 51.68% |
| F1-score | 64.14% |

The results suggest that JPEG compression augmentation changed the model's behaviour, including its ability to recognise real images.

However, the historical Model A and Model B evaluations were not established as identical controlled samples. Their scores should therefore be interpreted as exploratory results rather than a definitive comparison.

---

## 4. Model C — Expanded Dataset Experiment

Model C extends the original CIFAKE training experiment by combining CIFAKE with images from eight GenImage generator families.

The objective is to investigate whether exposure to a wider range of generated images improves performance beyond the original CIFAKE-only training setup.

### 4.1 Dataset composition

The combined training dataset contained 115,999 images.

| Dataset component | Images |
|---|---:|
| CIFAKE training images | 100,000 |
| GenImage images | 15,999 |
| Total combined dataset | 115,999 |
| AI-generated images | 58,006 |
| Real images | 57,993 |
| Training split | 104,400 |
| Validation split | 11,599 |
| Separate official CIFAKE test set | 20,000 |

The GenImage collection included eight generator families:

- ADM
- BigGAN
- GLIDE
- Midjourney
- Stable Diffusion 1.4
- Stable Diffusion 1.5
- VQDM
- Wukong

The official CIFAKE test set was kept separate from training and validation.

### 4.2 Training configuration

| Setting | Value |
|---|---:|
| CNN parameters | 889,633 |
| Image size | 64 × 64 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Training/validation split | 90% / 10% |
| Maximum epochs | 20 |
| Epochs completed | 12 |
| Early stopping patience | 5 |
| Training hardware | NVIDIA Tesla T4 |
| Precision | Mixed FP16/FP32 |

The experiment used the same custom CNN architecture, with AdamW optimisation and early stopping.

### 4.3 Training progress

The model completed 12 epochs before early stopping.

Epoch 7 produced the lowest validation loss and was selected as the best checkpoint.

| Metric | Epoch 7 |
|---|---:|
| Training accuracy | 93.70% |
| Validation accuracy | 92.40% |
| Training loss | 0.147297 |
| Validation loss | 0.178956 |

![Model C Training Accuracy](figures/model_c_accuracy.png)

*Figure 1. Training and validation accuracy across the completed epochs.*

![Model C Training Loss](figures/model_c_loss.png)

*Figure 2. Training and validation loss across the completed epochs.*

Training accuracy continued to increase after epoch 7, while validation loss reached its minimum at epoch 7. The checkpoint was therefore selected using validation loss rather than simply choosing the final epoch.

### 4.4 Official CIFAKE test results

The selected epoch 7 checkpoint was evaluated on the separate 20,000-image CIFAKE official test set.

| Metric | Model C |
|---|---:|
| Accuracy | **95.305%** |
| Precision | **93.777%** |
| Recall | **97.050%** |
| F1-score |
