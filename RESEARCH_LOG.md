# TruthScan - Pre-hackathon research log (written 2026-10-31, before event start)

Generic model research and evaluation only. No competition app code written yet.
AI tool disclosure: Claude (Anthropic) assisted with research, scripts and debugging.

## Tools prepared before the event
- Python 3.12 venv, PyTorch (CPU), Transformers, timm, Streamlit, FFmpeg
- Preprocessing test: FFmpeg frame extraction + 16 kHz mono audio extraction (synthetic test video)

## Image detector research
- UnivFD (CLIP ViT-L/14): 200-image CIFAKE sample, accuracy 40%, ROC-AUC 0.33 -> rejected.
- OwensLab/commfor-model-384 (MIT): runs, but missed 3 of 4 AI images on a tiny test.
- Selected: Thermostatic/community-forensics-frontier-detector-2026-08 (MIT, ViT-S/16 384, 21.8M params; fine-tune of OwensLab model).
  Informal check, 20 images (10 real phone/web photos, 10 AI incl. ChatGPT):
  - Default preprocessing: accuracy 75%, 5/10 real photos wrongly flagged AI, 10/10 AI caught, ROC-AUC 0.910
  - With "long edge to 768 px + JPEG q60" (described in the model card): accuracy 85%, 2/10 real wrongly flagged, 9/10 AI caught, ROC-AUC 0.950
  - Caveat: only 20 images; preprocessing choice was made after seeing first results on the same set. Not a reliable accuracy estimate.
  - Speed ~0.18 s/image, ~0.5 GB RAM on CPU (Intel Core Ultra 5 125H).

## Audio detector research
- Sadanie/pellav2-audio-deepfake-detector (Wav2Vec2-BERT 2.0 backbone, 2.32 GB checkpoint). Licence: MIT (checked on model card; card documents no training data or accuracy figures). Backbone facebook/w2v-bert-2.0 licence still to be checked.
- Test: 4 clips (2 real phone recordings, 2 TTS). 4/4 correct; also 4/4 after putting all clips through the same MP3 compression (format-confound control).
- Speed ~1.5-2.7 s per clip, ~5 GB RAM. Model analyses a 4-second centre crop only.
- Caveat: 4 clips cannot measure accuracy.

## Known limitations to state in README
- Scores are model scores, not calibrated probabilities. Detection is not proof of intent or misinformation.
- Image model is weak on very small / heavily compressed images and small edited regions (per model card).
- Real phone photos can be wrongly flagged as AI.

## Image detector, 100-image test (2026-10-31), web preprocessing
50 real, 50 AI (own photos, Wikimedia/Unsplash downloads, AI images from ChatGPT/Google and other tools). Accuracy 88%, 8/50 real called AI, 4/50 AI missed, ROC-AUC 0.953. On the 80 images collected after choosing the preprocessing: about 89%. Small sample; uncertainty roughly +/-10 points. Real photo real_b024 scored very high for AI (cause unknown).

## Audio detector hard test (2026-10-31)
PellaV2: 24 clips (12 real phone recordings, 12 clean TTS) scored 24/24, but this likely reflects recording channel (phone vs clean TTS). Harder test: 2 clean studio real clips -> both real; 3 TTS clips replayed through speakers and re-recorded on a phone -> all 3 called real (logits -6.7, -4.0, -6.5). Conclusion: detector does not reliably detect synthetic speech after acoustic replay. 5 clips only. Do not claim audio accuracy; state limitation in README.

## Audio degradation test (2026-10-31)
Same 24 clips (12 phone-recorded real, 12 clean TTS) after 300-3400 Hz band-limit + 16 kbps Opus: 24/24 correct, logits unchanged (fake ~+7.4, real ~-7.0). Robust to compression. Confound NOT resolved: real clips are all phone recordings, fakes are all clean TTS; replay test (3 clips) showed re-recorded TTS is called real. Claim only: detects clean digital synthetic speech; fails on over-the-air re-recording. Small sample.

## Video check, 2 clips (2026-10-31)
AI video (Golgappe-Ka-SnapYT.App.mp4, downloaded AI-generated clip, re-encoded by a downloader): 10/10 sampled frames above the image threshold (logits 3.1 to 10.4, median 5.6); audio windows (4 s) at -0.7 and -3.4, i.e. not flagged but much closer to the boundary than real clips (about -7). Real phone video (own recording with speech): 0/12 frames flagged (median -6.5), 0/4 audio windows flagged (about -7.0). Two videos only: shows pipeline works, not accuracy. Image evidence and audio evidence disagreed on the AI clip, so the report must show them separately.
