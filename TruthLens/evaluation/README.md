# Audio and video evaluation notes

The CSV manifests are exploratory inventories of the samples already present
in this repository. Every current row is marked `exploratory_seen`; these are
not final holdout measurements and must not be used to tune a production
threshold.

Run model-free audio conversion diagnostics with:

```powershell
.venv\Scripts\python.exe eval_audio_conditions.py --diagnostics-only
```

The script uses FFprobe and the production four-second center-window decode to
report source codec/rate/channels, decoded sample count, RMS/peak level, and
clipping. Running it without `--diagnostics-only` attempts the unchanged
PellaV2 production path at its existing score cutoff, but only after checking
for at least 5 GB available RAM. It reports counts and precision, recall, F1,
FPR, FNR, accuracy, and per-clip inference time. It does not tune a threshold.

Run video aggregation comparisons with:

```powershell
.venv\Scripts\python.exe eval_video_methods.py
```

That uses the production frame extraction and image detector path, compares
mean, median, and the fraction of sampled frames at/above the existing 0.50
frame score boundary, and emits visual-only metrics plus timings. The video
evaluation is guarded at 1.5 GB available RAM. No aggregate is a whole-video
authenticity verdict.

Recompute image threshold comparisons from the saved model logits without
loading the image model:

```powershell
.venv\Scripts\python.exe eval_image_thresholds.py
```

The tool treats AI-generated as the positive class and audits exact-byte
duplicates between the 100-image `test_images/` set and the 80-row follow-up
CSV's source folders. All 80 follow-up images are also present in the 100-image
set; count them once when reasoning about evidence. Neither set is a final
holdout.

## Data needed for a useful replay evaluation

Add fresh, held-out sessions with paired conditions: (1) genuine speech and
(2) synthetic speech played through a physical speaker and captured through
the same microphone/phone, room, distance, and recording settings. Include
several TTS generators, speakers, devices, rooms, background noise levels, and
lossy codecs. Keep source utterances and recording sessions grouped in one
split, so segments from one recording cannot leak across development and final
evaluation. Retain a separate final holdout; do not use it for threshold or
aggregation selection.

For video, gather multiple independently sourced genuine and generated clips
across durations, codecs, scene types, and generators. Split by source clip or
production session. Report both per-frame evidence statistics and per-video
metrics for any explicitly chosen aggregation rule; keep the visual metrics
separate from audio.
