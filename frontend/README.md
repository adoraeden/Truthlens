# TruthLens

TruthLens is a React/Vite interface backed by a Python FastAPI service for screening images, audio, and video for indicators of AI-generated media. It is a hackathon prototype for human review, not proof of authenticity or disinformation.

## Run the existing application

Start the Python backend in a terminal:

```powershell
Set-Location 'C:\Users\adora\DeepfakeLab'
.\.venv\Scripts\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8001
```

Start the web client in a second terminal:

```powershell
Set-Location 'C:\Users\adora\DeepfakeLab\frontend'
npm run dev
```

Open `http://localhost:3000`. Vite proxies `/api` requests to the backend at `http://127.0.0.1:8001`. The backend health check is `http://127.0.0.1:8001/api/health`. See the backend [README](../README.md) for model details, preprocessing, limitations, and API routes.

The Python environment needs PyTorch, torchvision, timm, transformers, safetensors, Hugging Face Hub, FastAPI, Pillow, NumPy, soundfile, torchaudio, SciPy, and FFmpeg/FFprobe. The current workspace has a backend `.venv`, local PellaV2 weights, and installed frontend `node_modules`. First-time model startup may need network access for Hugging Face model files.

## How to read results

- Image and audio screens show the detector’s model score and its own verdict. These scores are **not calibrated probabilities**.
- Audio analysis uses up to four seconds. Longer recordings use their center crop; shorter recordings are padded to the model input length. The result discloses the analyzed duration and source coverage.
- Video analysis evaluates sampled frame timestamps and, when present, a four-second center audio window separately. It does not produce an overall video probability. Sampled frames do not cover every moment.
- Image and sampled video-frame results show a positive input-gradient × input overlay for the exact inference crop. It is an approximate attribution, not proof of manipulation or a causal explanation. Audio does not return an attribution; the audio player is only a signal preview.

## Models, data, and disclosure

TruthLens uses published pretrained components and checkpoints: Community Forensics ViT-S/16 with the Thermostatic image checkpoint, and Sadanie/PellaV2 with the `facebook/w2v-bert-2.0` audio backbone. They were not trained from scratch for this app. The Python backend contains the inference implementations and checkpoint details.

The included image dataset is CIFAKE-derived (CIFAR-10 real images and Stable Diffusion 1.4 synthetic images). Existing CSVs and research notes record small exploratory checks; they are not independent benchmarks and should not be presented as general accuracy claims. Review upstream model and dataset documentation for provenance and license terms.

AI assistance disclosure: Claude supported pre-hackathon research, scripts, and debugging (recorded in the backend `RESEARCH_LOG.md`); Codex assisted with the current audit and reliability implementation.
