import sys
import math
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Allow importing the official Community Forensics architecture.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_MODEL_DIR = PROJECT_ROOT / "Community-Forensics"

if str(IMAGE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_MODEL_DIR))

import models


REPO = "Thermostatic/community-forensics-frontier-detector-2026-08"
LOGIT_THRESHOLD = 1.359375
MAX_IMAGE_PIXELS = 100_000_000


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

    def predict(self, image):
        try:
            if isinstance(image, Image.Image):
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the 100-megapixel analysis limit.")
                image.load()
                image = image.copy()
            else:
                with Image.open(image) as opened_image:
                    if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image dimensions exceed the 100-megapixel analysis limit.")
                    opened_image.load()
                    image = opened_image.copy()
        except (Image.UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The uploaded file is not a readable supported image.") from exc

        image = image.convert("RGB")

        with torch.inference_mode():
            tensor = self.model.preprocess_input(image)
            logit = self.model.vit(tensor).item()

        shifted_logit = logit - LOGIT_THRESHOLD
        if shifted_logit >= 0:
            score = 1 / (1 + math.exp(-shifted_logit))
        else:
            exp_logit = math.exp(shifted_logit)
            score = exp_logit / (1 + exp_logit)
        verdict = "AI-generated" if logit >= LOGIT_THRESHOLD else "Likely real"

        return {
            "verdict": verdict,
            "score": score,
            "logit": logit,
            "image_size": image.size,
            "model": "Community Forensics ViT-S/16",
            "analysis_note": (
                "The RGB image is resized to a 440-pixel shorter edge and center-cropped "
                "to 384 × 384; areas outside that crop are not analyzed. The model score "
                "is not a calibrated probability."
            ),
            "preprocessing": {
                "resize_short_edge": 440,
                "center_crop": [384, 384],
                "color_mode": "RGB",
            },
        }
