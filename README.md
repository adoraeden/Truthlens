# TruthLens media screening backend

TruthLens is a hackathon prototype that screens uploaded images, audio recordings, and videos for indicators associated with AI-generated media. It is an assistive review tool: model scores are uncalibrated, results can be wrong, and a result is not proof of authenticity or disinformation.

## Architecture and models

- **Images:** Community Forensics ViT-S/16, using the `Thermostatic/community-forensics-frontier-detector-2026-08` checkpoint. Images are converted to RGB, resized so the shorter edge is 440 pixels, center-cropped to 384 × 384, and ImageNet-normalized. Pixels outside the center crop are not analyzed. A fixed raw-logit cutoff of 1.359375 determines the verdict; the displayed model score is transformed so it crosses 50% at that same cutoff. It is not a calibrated probability.
- **Audio:** PellaV2 with a Wav2Vec2-BERT 2.0 backbone and a locally stored `models/pellav2/pellav2_detector.pt` checkpoint. FFmpeg converts audio to 16 kHz mono PCM. Clips up to four seconds are analyzed in full (short clips are zero-padded); longer clips use a four-second center crop. The model is intended for synthetic-speech screening, and its sigmoid score is not a calibrated probability.
- **Video:** FFprobe inspects the container. Up to eight frames are extracted at the midpoints of uniform time intervals and passed to the image detector. If audio exists, a center window of at most four seconds is sent to the audio detector. Image and audio findings remain separate. The mean sampled-frame score is a descriptive statistic, not a whole-video probability. Sampled timestamps do not represent every moment.

The project uses pretrained and externally published models; the models were **not trained from scratch for TruthLens**. The image checkpoint is fetched from Hugging Face on backend startup if it is not cached. PellaV2 weights are included locally; its Wav2Vec2-BERT backbone may also need its Hugging Face files cached. No model or dataset download is performed by the reliability changes in this project.

The exploratory image evaluator also has an optional 768-pixel long-edge resize and JPEG-quality-60 conversion path. The production detector does not use that transform. Existing small tests do not justify changing production preprocessing, and their results should not be treated as validation of the current production path.

## Run locally on Windows

The backend environment needs Python, PyTorch and the model/runtime packages used by the detector code, plus FFmpeg and FFprobe on `PATH`. This workspace already has a `.venv` and the PellaV2 checkpoint.

From `C:\Users\adora\DeepfakeLab`, start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000
```

In another terminal, start the React/Vite client:

```powershell
Set-Location 'C:\Users\adora\Downloads\truthlens---cyber-candy-ai-media-forensics'
npm run dev
```

Vite serves the UI on port 3000 and proxies `/api` to `http://127.0.0.1:8000`. The API health endpoint is `http://127.0.0.1:8000/api/health`. The first backend startup can take time while model weights load; image checkpoint access requires a prior cache or network access.

## API

All endpoints accept one multipart upload field named `file`, enforce a 100 MiB upload limit, and preserve the existing response envelope (`success`, `modality`, `filename`, `result`).

- `GET /api/health`
- `POST /api/detect/image` — `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- `POST /api/detect/audio` — `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`
- `POST /api/detect/video` — `.mp4`, `.mov`, `.mkv`, `.avi`

Invalid or undecodable media receives a client error with a user-facing message. Other inference failures are logged by the server and return a generic message without exposing internal paths or stack traces. Inference runs in FastAPI's thread pool; a lock serializes access to the shared model instances.

## Evaluation data and limitations

`data/train.zip` and the extracted `data/kaggle_image` folders contain CIFAKE-style image data. The source data pairs CIFAR-10 real images with Stable Diffusion 1.4 fakes and is not a reliable measure of performance on modern generators or high-resolution media. `eval_results*.csv`, `eval_audio_results.csv`, and `RESEARCH_LOG.md` contain informal exploratory checks, not independent benchmark results. The audio notes document recording-channel confounds and missed replayed synthetic speech. The video notes cover only a couple of example clips. Do not present these as general accuracy claims.

There is no validated image heatmap, audio explanation, lip-sync comparison, or fused whole-video verdict. Video analysis may miss manipulations between sampled frames; audio analysis may miss content outside the selected window. Scores from all modalities are model outputs, not calibrated probabilities.

The API runs inference in a thread pool so media processing does not run on the async event loop, but a process-level lock serializes inference through the shared model instances. This prototype does not include a distributed worker queue and has not been load-tested for high-volume processing.

## Data, checkpoints, and AI assistance disclosure

- Image model: Community Forensics architecture and the published Thermostatic checkpoint; see `Community-Forensics/LICENSE` and the upstream model card for current terms and provenance.
- Audio model: Sadanie/PellaV2 checkpoint and `facebook/w2v-bert-2.0` backbone; see `models/pellav2/README.md` and upstream model documentation for terms and provenance.
- Image data: CIFAKE (real CIFAR-10 images and Stable Diffusion 1.4 synthetic images); see `data/DATASET_NOTES.txt` and `dataset-metadata.json`.
- Development tools: Claude assisted pre-hackathon research, scripts, and debugging as recorded in `RESEARCH_LOG.md`. Codex assisted with the TruthLens audit and reliability implementation. No claim is made that the detectors were trained from scratch.
