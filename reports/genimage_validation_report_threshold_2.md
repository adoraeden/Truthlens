# TruthLens GenImage Validation Report

## Evaluation setup

- Dataset: `data/genimage_validation_sample` (manifest labels; AI is positive, nature is negative).
- Production decision threshold: raw logit ≥ 2.
- Detector: existing Community Forensics ViT-S/16 via `ImageDetector.predict`; no model or preprocessing changes.
- `threshold_relative_score_uncalibrated` is sigmoid(logit − threshold), not a calibrated probability.
- Calibrated probabilities are not used for predictions or reported as evaluation scores.

## Overall metrics

### Overall
Samples: 400 total, 400 scored, 0 errors; coverage 1.0000
Class counts (total): AI=200, nature=200; scored: AI=200, nature=200; errors: AI=0, nature=0
Confusion (AI positive): TP=148, FP=2, TN=198, FN=52
Accuracy=0.8650; precision=0.9867; recall=0.7400; specificity=0.9900; F1=0.8457; balanced accuracy=0.8650; FPR=0.0100; FNR=0.2600

## Per-generator metrics

### ADM
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=6, FP=0, TN=25, FN=19
Accuracy=0.6200; precision=1.0000; recall=0.2400; specificity=1.0000; F1=0.3871; balanced accuracy=0.6200; FPR=0.0000; FNR=0.7600

### BigGAN
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=11, FP=0, TN=25, FN=14
Accuracy=0.7200; precision=1.0000; recall=0.4400; specificity=1.0000; F1=0.6111; balanced accuracy=0.7200; FPR=0.0000; FNR=0.5600

### glide
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=23, FP=0, TN=25, FN=2
Accuracy=0.9600; precision=1.0000; recall=0.9200; specificity=1.0000; F1=0.9583; balanced accuracy=0.9600; FPR=0.0000; FNR=0.0800

### Midjourney
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=21, FP=1, TN=24, FN=4
Accuracy=0.9000; precision=0.9545; recall=0.8400; specificity=0.9600; F1=0.8936; balanced accuracy=0.9000; FPR=0.0400; FNR=0.1600

### stable_diffusion_v_1_4
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=25, FP=1, TN=24, FN=0
Accuracy=0.9800; precision=0.9615; recall=1.0000; specificity=0.9600; F1=0.9804; balanced accuracy=0.9800; FPR=0.0400; FNR=0.0000

### stable_diffusion_v_1_5
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=25, FP=0, TN=25, FN=0
Accuracy=1.0000; precision=1.0000; recall=1.0000; specificity=1.0000; F1=1.0000; balanced accuracy=1.0000; FPR=0.0000; FNR=0.0000

### VQDM
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=12, FP=0, TN=25, FN=13
Accuracy=0.7400; precision=1.0000; recall=0.4800; specificity=1.0000; F1=0.6486; balanced accuracy=0.7400; FPR=0.0000; FNR=0.5200

### wukong
Samples: 50 total, 50 scored, 0 errors; coverage 1.0000
Class counts (total): AI=25, nature=25; scored: AI=25, nature=25; errors: AI=0, nature=0
Confusion (AI positive): TP=25, FP=0, TN=25, FN=0
Accuracy=1.0000; precision=1.0000; recall=1.0000; specificity=1.0000; F1=1.0000; balanced accuracy=1.0000; FPR=0.0000; FNR=0.0000

## Exact-hash overlap with local prior evaluation images

Indexed 189 files under `test_images, new_real, new_fake, mistake_review`. Exact GenImage hash overlaps: **0**.
- No byte-identical sample hashes were found in those local evaluation-image directories.

## Processing validation

Successfully scored 400 of 400 manifest samples.
All manifest samples were processed successfully.
