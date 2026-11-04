import sys
import math
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from .image_diagnostics import (
    build_visualizations,
    collect_source_metadata,
    compute_attribution_heatmap,
    png_base64,
)

# Allow importing the official Community Forensics architecture.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_MODEL_DIR = PROJECT_ROOT / "Community-Forensics"

if str(IMAGE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_MODEL_DIR))

import models


REPO = "Thermostatic/community-forensics-frontier-detector-2026-08"


def _configured_image_logit_threshold():
    raw_value = os.environ.get("TRUTHLENS_IMAGE_LOGIT_THRESHOLD", "0.0")
    try:
        threshold = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TRUTHLENS_IMAGE_LOGIT_THRESHOLD must be a finite number") from exc
    if not math.isfinite(threshold):
        raise RuntimeError("TRUTHLENS_IMAGE_LOGIT_THRESHOLD must be a finite number")
    return threshold


LOGIT_THRESHOLD = _configured_image_logit_threshold()
SCORE_CUTOFF = 0.5
MAX_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_BYTES = 100 * 1024 * 1024


class ImageDetector:
    def __init__(self):
        torch.set_num_threads(8)

        args = type("Args", (), {
            "model_size": "small",
            "patch_size": 16,
            "freeze_backbone": False,
            "input_size": 384,
        })()

        self.model = models.ViTClassifier(args, device="cpu")

        checkpoint_path = hf_hub_download(
            repo_id=REPO,
            filename="model.safetensors",
        )

        state_dict = load_file(checkpoint_path)
        state_dict = {
            ("vit." + key if not key.startswith("vit.") else key)
            .replace("module.", ""): value
            for key, value in state_dict.items()
        }

        result = self.model.load_state_dict(state_dict, strict=False)

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing: {result.missing_keys}; "
                f"Unexpected: {result.unexpected_keys}"
            )

        self.model.eval()
        from .image_calibration import load_calibrator
        self.calibrator = load_calibrator()

    @staticmethod
    def _prepare_web_view(image):
        """Apply the checkpoint's documented 768px/JPEG-60 web evaluation view."""
        width, height = image.size
        scale = 768 / max(width, height)
        if scale < 1:
            image = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=60, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    def predict(self, image):
        try:
            if isinstance(image, Image.Image):
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the 100-megapixel analysis limit.")
                image.load()
                source_metadata = collect_source_metadata(image)
                image = image.copy()
            else:
                image_path = Path(image) if isinstance(image, (str, Path)) else None
                if image_path is not None and image_path.is_file() and image_path.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("Image file exceeds the 100 MB analysis limit.")
                with Image.open(image) as opened_image:
                    if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image dimensions exceed the 100-megapixel analysis limit.")
                    opened_image.load()
                    source_metadata = collect_source_metadata(opened_image)
                    image = opened_image.copy()
        except (Image.UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The uploaded file is not a readable supported image.") from exc

        source_size = image.size
        image = ImageOps.exif_transpose(image)
        source_image = image.convert("RGB")
        image = self._prepare_web_view(source_image)

        # The model receives its unchanged production view and preprocessing.
        # Reuse this one forward pass for the existing input-gradient attribution.
        tensor = self.model.preprocess_input(image).detach().requires_grad_(True)
        with torch.enable_grad():
            # ViTClassifier.forward applies sigmoid; use its one-unit `vit`
            # head output here because this pipeline thresholds raw logits.
            logits = self.model.vit(tensor).reshape(-1)
            if logits.numel() != 1:
                raise RuntimeError(
                    f"Expected one image logit, received {logits.numel()} outputs."
                )
            logit_tensor = logits[0]
            if not torch.isfinite(logit_tensor):
                raise ValueError("The image detector returned a non-finite logit.")
            gradient, = torch.autograd.grad(logit_tensor, tensor)
        logit = logit_tensor.detach().item()

        # Positive gradient × input is an approximate input-attribution map,
        # not a causal explanation. It corresponds to the actual model tensor.
        attribution = torch.relu((gradient * tensor).sum(dim=1))[0]
        attribution = attribution.detach().cpu().numpy()
        attribution = np.nan_to_num(attribution, nan=0.0, posinf=0.0, neginf=0.0)
        # Recover the visible crop from the exact normalized tensor that was
        # scored. This keeps map and crop pixels spatially aligned by design.
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_pixels = (
            (tensor.detach().cpu()[0] * std.cpu() + mean.cpu())
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        crop = Image.fromarray(crop_pixels, mode="RGB")
        heatmap, spatial_attribution = compute_attribution_heatmap(attribution)
        normalized = attribution / float(attribution.max()) if float(attribution.max()) > 0 else np.zeros_like(attribution)
        heat_pixels = np.asarray(heatmap, dtype=np.float32)
        alpha = (normalized[..., None] * 0.65).astype(np.float32)
        overlay_pixels = np.clip(crop_pixels.astype(np.float32) * (1 - alpha) + heat_pixels * alpha, 0, 255)
        overlay = Image.fromarray(overlay_pixels.astype(np.uint8), mode="RGB")
        visualizations = build_visualizations(
            source_image,
            attribution,
            source_metadata,
            heatmap=heatmap,
            spatial=spatial_attribution,
        )
        explanation = {
            "method": "positive input-gradient × input",
            "scope": "384 × 384 resized center crop used for inference",
            "note": "Approximate positive input-gradient × input attribution. It is not Grad-CAM, causal proof, or confirmation of manipulation.",
            "overlay_png_base64": png_base64(overlay),
            "crop_png_base64": png_base64(crop),
            **visualizations,
        }

        shifted_logit = logit - LOGIT_THRESHOLD
        if shifted_logit >= 0:
            score = 1 / (1 + math.exp(-shifted_logit))
        else:
            exp_logit = math.exp(shifted_logit)
            score = exp_logit / (1 + exp_logit)
        verdict = "AI-generated" if logit >= LOGIT_THRESHOLD else "Likely real"
        from .image_calibration import calibrated_probabilities
        calibration = calibrated_probabilities(logit, getattr(self, "calibrator", None))

        return {
            "positive_class": "AI-generated",
            "verdict": verdict,
            "score": score,
            "logit": logit,
            "decision_threshold": LOGIT_THRESHOLD,
            "score_cutoff": SCORE_CUTOFF,
            "score_method": "sigmoid(logit - decision_threshold)",
            "score_calibrated_probability": False,
            "calibration_status": "available" if calibration is not None else "unavailable",
            "calibration_method": self.calibrator.get("method") if calibration is not None else None,
            "ai_generated_probability": calibration["ai_generated_probability"] if calibration is not None else None,
            "genuine_probability": calibration["genuine_probability"] if calibration is not None else None,
            "calibration_note": (
                "Separately calibrated estimate; it does not change the threshold-based verdict."
                if calibration is not None
                else "Calibration unavailable: no validated production calibrator is installed."
            ),
            "image_size": source_size,
            "model": "Community Forensics ViT-S/16",
            "analysis_note": (
                "The RGB image is downscaled to a maximum 768-pixel long edge and JPEG "
                "quality 60, then resized to a 440-pixel shorter edge and center-cropped "
                "to 384 × 384. Areas outside that crop are not analyzed. The model score "
                "is not a calibrated probability."
            ),
            "explanation": explanation,
            "preprocessing": {
                "resize_short_edge": 440,
                "center_crop": [384, 384],
                "color_mode": "RGB",
            },
        }
