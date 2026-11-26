
import logging
import shutil
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .image_detector import ImageDetector
from .audio_detector import AudioDetector, available_memory_gb, pella_can_load, PELLA_MIN_AVAILABLE_GB
from .video_detector import VideoDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlens")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"},
    "video": {".mp4", ".mov", ".mkv", ".avi"},
}

VIDEO_PREVIEW_CONVERT_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

# Audio extensions whose containers browsers cannot reliably decode natively
# (AAC-in-M4A, FLAC, OGG with Opus, raw AAC). We offer a preview-conversion
# endpoint that re-muxes them into an MP3 the browser can always play.
AUDIO_PREVIEW_CONVERT_EXTENSIONS = {".m4a", ".flac", ".aac", ".ogg"}

detectors = {
    "image": None,
    "audio": None,
    "video": None,
}
model_lock = threading.Lock()
audio_unavailable_reason = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global audio_unavailable_reason
    logger.info("Loading TruthLens detectors...")

    # Measure available RAM *before* loading any model so that the image
    # detector's footprint does not cause the audio check to under-count.
    available_gb_at_startup = available_memory_gb()
    logger.info("Available RAM at startup: %.2f GB (minimum for PellaV2: %.1f GB)",
                available_gb_at_startup, PELLA_MIN_AVAILABLE_GB)

    try:
        detectors["image"] = ImageDetector()
        if not pella_can_load(available_gb_at_startup):
            audio_unavailable_reason = (
                f"PellaV2 requires at least {PELLA_MIN_AVAILABLE_GB:.1f} GB of free RAM to load, "
                f"but only {available_gb_at_startup:.2f} GB was available when the server started. "
                f"Close other applications to free memory, then restart the backend. "
                f"You can lower the threshold by setting the environment variable "
                f"TRUTHLENS_PELLA_MIN_RAM_GB (current default: {PELLA_MIN_AVAILABLE_GB:.1f})."
            )
            detectors["audio"] = None
            logger.warning("PellaV2 not loaded: %.2f GB available, %.1f GB required.",
                           available_gb_at_startup, PELLA_MIN_AVAILABLE_GB)
        else:
            try:
                detectors["audio"] = AudioDetector()
                audio_unavailable_reason = None
            except Exception as exc:
                detectors["audio"] = None
                audio_unavailable_reason = str(exc) or "PellaV2 failed to initialize; audio analysis is unavailable."
                logger.exception("Audio detector initialization failed: %s", exc)
        detectors["video"] = VideoDetector(
            detectors["image"],
            detectors["audio"],
            max_frames=8,
            audio_unavailable_reason=audio_unavailable_reason,
        )
        logger.info("TruthLens visual detectors are ready.")
    except Exception:
        logger.exception("Detector initialization failed.")
        raise

    yield

    detectors.update(image=None, audio=None, video=None)
    audio_unavailable_reason = None
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
def health():
    ready = detectors["image"] is not None
    return {
        "status": "online" if ready else "offline",
        "service": "TruthLens Detector API",
        "version": "2.4.0",
        "detectors": {
            name: detector is not None
            for name, detector in detectors.items()
        },
        "audio_unavailable_reason": audio_unavailable_reason,
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
            detail = (
                audio_unavailable_reason
                if modality == "audio" and audio_unavailable_reason
                else f"The {modality} detector is not ready."
            )
            raise HTTPException(
                status_code=503,
                detail=detail,
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


@app.post("/api/convert/audio-preview")
async def convert_audio_preview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Convert an uploaded audio file to a browser-compatible MP3 for in-page
    preview playback.  This endpoint is *only* used for the visual player;
    the original file is always submitted separately to PellaV2 for forensic
    analysis and is never replaced by the converted preview.

    Returns the MP3 as a streaming response.  The temporary source and output
    files are deleted after the response body is sent.
    """
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS["audio"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio file format for preview conversion: {suffix or 'unknown'}",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 100 MB upload limit.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is not available on the server; audio preview conversion is unavailable.",
        )

    # Write original to a temp file so FFmpeg can read it
    src_tmp = tempfile.NamedTemporaryFile(
        prefix="tl_preview_src_",
        suffix=suffix,
        delete=False,
    )
    src_path = Path(src_tmp.name)
    try:
        with src_tmp:
            src_tmp.write(data)
    except Exception:
        src_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to write temporary source file.")

    out_path = src_path.with_suffix(".mp3")

    def _do_convert():
        try:
            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-v", "error",
                    "-i", str(src_path),
                    "-vn",           # strip any video stream
                    "-ac", "2",      # stereo (browsers handle stereo better)
                    "-ar", "44100",  # standard web sample rate
                    "-b:a", "128k",  # reasonable preview bitrate
                    str(out_path),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
        finally:
            src_path.unlink(missing_ok=True)

    try:
        await run_in_threadpool(_do_convert)
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        logger.warning("Audio preview conversion failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Audio preview conversion failed: {exc}",
        ) from exc

    if not out_path.is_file() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="FFmpeg produced no output during audio preview conversion.",
        )

    # Schedule cleanup after the response is fully sent
    background_tasks.add_task(out_path.unlink, missing_ok=True)

    return FileResponse(
        path=str(out_path),
        media_type="audio/mpeg",
        filename=Path(filename).stem + ".mp3",
    )


@app.post("/api/convert/video-preview")
async def convert_video_preview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Convert an uploaded video file to a browser-compatible H.264 MP4 with yuv420p
    pixel format and faststart for in-page preview playback.
    This endpoint is *only* used for the browser preview player; the original
    file is always submitted separately to the forensic analysis pipeline and is
    never replaced by the converted preview.
    """
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS["video"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video file format for preview conversion: {suffix or 'unknown'}",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 100 MB upload limit.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is not available on the server; video preview conversion is unavailable.",
        )

    src_tmp = tempfile.NamedTemporaryFile(
        prefix="tl_vid_preview_src_",
        suffix=suffix,
        delete=False,
    )
    src_path = Path(src_tmp.name)
    try:
        with src_tmp:
            src_tmp.write(data)
    except Exception:
        src_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to write temporary source file.")

    out_path = src_path.with_name(f"{src_path.stem}_browser.mp4")

    def _do_convert_video():
        try:
            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-v", "error",
                    "-i", str(src_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-max_muxing_queue_size", "1024",
                    str(out_path),
                ],
                capture_output=True,
                timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
        finally:
            src_path.unlink(missing_ok=True)

    try:
        await run_in_threadpool(_do_convert_video)
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        logger.warning("Video preview conversion failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Video preview conversion failed: {exc}",
        ) from exc

    if not out_path.is_file() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="FFmpeg produced no output during video preview conversion.",
        )

    background_tasks.add_task(out_path.unlink, missing_ok=True)

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=Path(filename).stem + "_preview.mp4",
    )
