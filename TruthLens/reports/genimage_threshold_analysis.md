# GenImage Exploratory Threshold Analysis

- Source: `genimage_validation_predictions.csv`; 400 saved prediction rows. No model inference was run.
- Ground truth: `ai` is positive; `nature` is negative.
- Current production raw-logit threshold: **2** (included in the sweep).
- This sweep is exploratory on the same evaluation sample. It does not independently validate any threshold and does not change production behavior.
- `threshold_relative_score_uncalibrated` is not a probability; the sweep uses raw logits only.

## Current-threshold metrics

Scored 400/400 rows; inference errors: 0.
Class counts: AI=200 (scored 200), nature=200 (scored 200).
TP=148, FP=2, TN=198, FN=52.
Accuracy=0.8650; precision=0.9867; recall=0.7400; specificity=0.9900; F1=0.8457; balanced accuracy=0.8650; FPR=0.0100; FNR=0.2600.

## Generator recall and false-positive rate at the current threshold

| Generator | AI count | Nature count | Recall | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADM | 25 | 25 | 0.2400 | 0.0000 | 6 | 0 | 25 | 19 |
| BigGAN | 25 | 25 | 0.4400 | 0.0000 | 11 | 0 | 25 | 14 |
| glide | 25 | 25 | 0.9200 | 0.0000 | 23 | 0 | 25 | 2 |
| Midjourney | 25 | 25 | 0.8400 | 0.0400 | 21 | 1 | 24 | 4 |
| stable_diffusion_v_1_4 | 25 | 25 | 1.0000 | 0.0400 | 25 | 1 | 24 | 0 |
| stable_diffusion_v_1_5 | 25 | 25 | 1.0000 | 0.0000 | 25 | 0 | 25 | 0 |
| VQDM | 25 | 25 | 0.4800 | 0.0000 | 12 | 0 | 25 | 13 |
| wukong | 25 | 25 | 1.0000 | 0.0000 | 25 | 0 | 25 | 0 |

## Current-threshold false negatives

Sorted by raw logit nearest to the boundary first.
- `VQDM/ai/VQDM_1000_200_03_312_vqdm_00127.png` (VQDM): logit 1.85589; 0.144113 below threshold
- `glide/ai/GLIDE_1000_200_09_981_glide_00035.png` (glide): logit 1.74169; 0.258309 below threshold
- `ADM/ai/497_adm_34.PNG` (ADM): logit 1.68274; 0.317264 below threshold
- `BigGAN/ai/552_biggan_00127.png` (BigGAN): logit 1.65848; 0.341517 below threshold
- `BigGAN/ai/932_biggan_00020.png` (BigGAN): logit 1.58967; 0.410335 below threshold
- `VQDM/ai/VQDM_1000_200_03_327_vqdm_00035.png` (VQDM): logit 1.48913; 0.510874 below threshold
- `ADM/ai/550_adm_34.PNG` (ADM): logit 1.37064; 0.629359 below threshold
- `BigGAN/ai/886_biggan_00143.png` (BigGAN): logit 1.22547; 0.77453 below threshold
- `glide/ai/GLIDE_1000_200_00_009_glide_00020.png` (glide): logit 1.17267; 0.82733 below threshold
- `BigGAN/ai/773_biggan_00020.png` (BigGAN): logit 1.16256; 0.837444 below threshold
- `ADM/ai/417_adm_85.PNG` (ADM): logit 0.964929; 1.03507 below threshold
- `ADM/ai/565_adm_34.PNG` (ADM): logit 0.957099; 1.0429 below threshold
- `Midjourney/ai/748_midjourney_169.png` (Midjourney): logit 0.682686; 1.31731 below threshold
- `BigGAN/ai/927_biggan_00143.png` (BigGAN): logit 0.620128; 1.37987 below threshold
- `BigGAN/ai/864_biggan_00127.png` (BigGAN): logit 0.311471; 1.68853 below threshold
- `ADM/ai/233_adm_7.PNG` (ADM): logit 0.306945; 1.69306 below threshold
- `Midjourney/ai/506_midjourney_169.png` (Midjourney): logit 0.160427; 1.83957 below threshold
- `VQDM/ai/VQDM_1000_200_04_475_vqdm_00143.png` (VQDM): logit 0.0348205; 1.96518 below threshold
- `ADM/ai/750_adm_174.PNG` (ADM): logit -0.435902; 2.4359 below threshold
- `BigGAN/ai/314_biggan_00127.png` (BigGAN): logit -0.46095; 2.46095 below threshold
- `ADM/ai/601_adm_34.PNG` (ADM): logit -0.511677; 2.51168 below threshold
- `VQDM/ai/VQDM_1000_200_01_187_vqdm_00074.png` (VQDM): logit -0.753808; 2.75381 below threshold
- `BigGAN/ai/692_biggan_00143.png` (BigGAN): logit -0.864178; 2.86418 below threshold
- `Midjourney/ai/770_midjourney_100.png` (Midjourney): logit -0.979198; 2.9792 below threshold
- `VQDM/ai/VQDM_1000_200_09_982_vqdm_00020.png` (VQDM): logit -1.07527; 3.07527 below threshold
- `VQDM/ai/VQDM_1000_200_09_988_vqdm_00074.png` (VQDM): logit -1.38893; 3.38893 below threshold
- `VQDM/ai/VQDM_1000_200_02_206_vqdm_00143.png` (VQDM): logit -1.52106; 3.52106 below threshold
- `ADM/ai/968_adm_91.PNG` (ADM): logit -1.59321; 3.59321 below threshold
- `VQDM/ai/VQDM_1000_200_04_478_vqdm_00035.png` (VQDM): logit -1.85264; 3.85264 below threshold
- `ADM/ai/16_adm_174.PNG` (ADM): logit -1.93413; 3.93413 below threshold
- `ADM/ai/748_adm_91.PNG` (ADM): logit -2.14208; 4.14208 below threshold
- `Midjourney/ai/324_midjourney_88.png` (Midjourney): logit -2.34095; 4.34095 below threshold
- `BigGAN/ai/395_biggan_00143.png` (BigGAN): logit -2.36767; 4.36767 below threshold
- `ADM/ai/608_adm_91.PNG` (ADM): logit -2.5482; 4.5482 below threshold
- `ADM/ai/473_adm_34.PNG` (ADM): logit -2.87172; 4.87172 below threshold
- `VQDM/ai/VQDM_1000_200_00_077_vqdm_00143.png` (VQDM): logit -2.88129; 4.88129 below threshold
- `ADM/ai/957_adm_91.PNG` (ADM): logit -3.40663; 5.40663 below threshold
- `BigGAN/ai/355_biggan_00039.png` (BigGAN): logit -3.49815; 5.49815 below threshold
- `ADM/ai/121_adm_153.PNG` (ADM): logit -3.50431; 5.50431 below threshold
- `BigGAN/ai/813_biggan_00143.png` (BigGAN): logit -3.76539; 5.76539 below threshold
- `ADM/ai/713_adm_91.PNG` (ADM): logit -3.76941; 5.76941 below threshold
- `VQDM/ai/VQDM_1000_200_01_121_vqdm_00127.png` (VQDM): logit -3.81769; 5.81769 below threshold
- `BigGAN/ai/043_biggan_00020.png` (BigGAN): logit -3.89123; 5.89123 below threshold
- `VQDM/ai/VQDM_1000_200_08_843_vqdm_00143.png` (VQDM): logit -3.95839; 5.95839 below threshold
- `VQDM/ai/VQDM_1000_200_02_261_vqdm_00127.png` (VQDM): logit -4.02776; 6.02776 below threshold
- `ADM/ai/389_adm_7.PNG` (ADM): logit -4.10192; 6.10192 below threshold
- `ADM/ai/532_adm_91.PNG` (ADM): logit -4.73799; 6.73799 below threshold
- `BigGAN/ai/752_biggan_00020.png` (BigGAN): logit -5.35122; 7.35122 below threshold
- `BigGAN/ai/641_biggan_00039.png` (BigGAN): logit -6.0904; 8.0904 below threshold
- `ADM/ai/441_adm_85.PNG` (ADM): logit -6.50306; 8.50306 below threshold
- `ADM/ai/443_adm_91.PNG` (ADM): logit -6.84064; 8.84064 below threshold
- `VQDM/ai/VQDM_1000_200_02_208_vqdm_00039.png` (VQDM): logit -8.55878; 10.5588 below threshold

## Current-threshold false positives

Sorted by raw logit nearest to the boundary first.
- `Midjourney/nature/ILSVRC2012_val_00041063.JPEG` (Midjourney): logit 3.58446; 1.58446 above threshold
- `stable_diffusion_v_1_4/nature/ILSVRC2012_val_00021522.JPEG` (stable_diffusion_v_1_4): logit 3.61341; 1.61341 above threshold

## Recall / false-positive trade-off

Lowering the threshold can classify more AI samples as positive (higher recall), while also classifying more nature samples as positive (higher FPR). Raising it generally moves in the opposite direction. These are empirical trade-offs on this sample, not a basis for automatically changing production.
At the adjacent lower threshold 1.75, recall is 0.7450 and FPR is 0.0100; at the current threshold 2, recall is 0.7400 and FPR is 0.0100.

No threshold is selected or applied by this analysis.
