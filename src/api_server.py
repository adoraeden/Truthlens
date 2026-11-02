
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from .image_detector import ImageDetector
from .audio_detector import AudioDetector
from .video_detector import VideoDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlens")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"},
    "video": {".mp4", ".mov", ".mkv", ".avi"},
}

detectors = {
    "image": None,
    "audio": None,
    "video": None,
}
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TruthLens detectors...")

    try:
        detectors["image"] = ImageDetector()
        detectors["audio"] = AudioDetector()
        detectors["video"] = VideoDetector(
            detectors["image"],
            detectors["audio"],
            max_frames=8,
        )
        logger.info("All TruthLens detectors are ready.")
    except Exception:
        logger.exception("Detector initialization failed.")
        raise

    yield

    detectors.update(image=None, audio=None, video=None)
    logger.info("TruthLens detector service stopped.")


app = FastAPI(
    title="TruthLens Detector API",
    version="2.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    ready = all(detectors.values())
    return {
        "status": "online" if ready else "offline",
        "service": "TruthLens Detector API",
        "version": "2.4.0",
        "detectors": {
            name: detector is not None
            for name, detector in detectors.items()
        },
    }


async def save_upload(upload: UploadFile, modality: str):
    filename = Path(upload.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS[modality]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported {modality} file format: {suffix or 'unknown'}",
        )

    data = await upload.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 100 MB upload limit.",
        )

    temp = tempfile.NamedTemporaryFile(
        prefix="truthlens_",
        suffix=suffix,
        delete=False,
    )
    path = Path(temp.name)

    try:
        with temp:
            temp.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return path


async def analyze_upload(upload: UploadFile, modality: str):
    path = None
    try:
        path = await save_upload(upload, modality)
        if detectors[modality] is None:
            raise HTTPException(
                status_code=503,
                detail=f"The {modality} detector is not ready.",
            )

        def run_inference():
            with model_lock:
                if modality == "image":
                    return detectors["image"].predict(path)

                if modality == "audio":
                    return detectors["audio"].predict(path)

                return detectors["video"].predict(path)

        result = await run_in_threadpool(run_inference)

        # Convert image dimensions from tuple to JSON-compatible list.
        if modality == "image" and "image_size" in result:
            result["image_size"] = list(result["image_size"])

        return {
            "success": True,
            "modality": modality,
            "filename": Path(upload.filename or "upload").name,
            "result": result,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("%s analysis failed", modality)
        raise HTTPException(
            status_code=500,
            detail=f"{modality.capitalize()} analysis failed. Check that the file is valid and try again.",
        ) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
        await upload.close()


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    return await analyze_upload(file, "image")


@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    return await analyze_upload(file, "audio")


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    return await analyze_upload(file, "video")
