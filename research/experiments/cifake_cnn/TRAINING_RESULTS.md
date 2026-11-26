# CIFAKE CNN — GPU Training Results

## 1. Experiment overview

The custom CNN was trained from random initialization using the CIFAKE dataset on Google Colab with GPU acceleration.

| Setting | Value |
|---|---|
| Dataset | CIFAKE |
| Training device | NVIDIA Tesla T4 |
| Framework | PyTorch 2.14.1+cu130 |
| GPU memory | Approximately 14.56 GB |
| Input resolution | 64 × 64 |
| Batch size | 64 |
| Maximum epochs | 20 |
| Early stopping patience | 5 |
| Random seed | 42 |
| Optimizer | AdamW |
| Loss function | BCEWithLogitsLoss |

## 2. Dataset split

The official CIFAKE training directory was divided into training and validation subsets.

| Split | Images |
|---|---:|
| Training | 90,000 |
| Validation | 10,000 |
| Official test | 20,000 |

The official test directory was kept separate and was not used for training or validation.

## 3. Training outcome

Training stopped after 12 epochs because of early stopping.

The best checkpoint was obtained at epoch 6.

| Metric | Best result |
|---|---:|
| Best epoch | 6 |
| Training loss | 0.0603 |
| Training accuracy | 97.71% |
| Validation loss | 0.1236 |
| Validation accuracy | 95.63% |

The best checkpoint was restored after training.

## 4. Training history

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.2573 | 89.14% | 0.1908 | 92.79% |
| 1 | 0.1577 | 93.89% | 0.1712 | 93.28% |
| 2 | 0.1281 | 95.08% | 0.1250 | 95.29% |
| 3 | 0.1083 | 95.94% | 0.1474 | 94.67% |
| 4 | 0.0909 | 96.53% | 0.1274 | 95.36% |
| 5 | 0.0760 | 97.14% | 0.1315 | 95.18% |
| 6 | 0.0603 | 97.71% | 0.1236 | 95.63% |
| 7 | 0.0478 | 98.15% | 0.1541 | 94.96% |
| 8 | 0.0407 | 98.46% | 0.1563 | 95.19% |
| 9 | 0.0348 | 98.73% | 0.1663 | 94.90% |
| 10 | 0.0302 | 98.91% | 0.1500 | 95.50% |
| 11 | 0.0272 | 98.99% | 0.1887 | 95.13% |

## 5. Observations

- Training loss decreased throughout the experiment.
- Validation accuracy reached its highest recorded value at epoch 6.
- Training accuracy continued improving after epoch 6, while validation loss generally increased.
- This suggests overfitting after the best checkpoint, which supports retaining the epoch 6 checkpoint rather than the final epoch.

## 6. Saved artifacts

The following artifacts were saved to Google Drive:

- Best model checkpoint: `checkpoints/cifake_cnn/best_cifake_cnn.pt`
- Training history: `training-logs/cifake_cnn/training_history.csv`

These are experiment artifacts and are not included in the Git repository.

## 7. Conclusion

The custom CNN successfully completed GPU training on CIFAKE and achieved 95.63% validation accuracy.

The epoch 6 checkpoint was selected using validation loss. This result demonstrates strong performance on the CIFAKE validation split, but does not establish generalisation to other datasets. Independent evaluation is required before drawing broader conclusions.
