"""Fail-closed loading and application of a separately validated image calibrator."""

import json
import math
from pathlib import Path

DEFAULT_CALIBRATOR_PATH = Path(__file__).resolve().parent.parent / "models" / "image_calibrator.json"
CALIBRATOR_SCHEMA_VERSION = 1


def load_calibrator(path=DEFAULT_CALIBRATOR_PATH):
    """Load only an explicitly production-eligible artifact matching this detector."""
    from .image_detector import LOGIT_THRESHOLD

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            data.get("schema_version") != CALIBRATOR_SCHEMA_VERSION
            or data.get("production_eligible") is not True
            or data.get("decision_threshold") != LOGIT_THRESHOLD
            or not data.get("dataset_version")
            or data.get("method") not in {"platt", "temperature"}
        ):
            return None
        parameters = data.get("parameters", {})
        if data["method"] == "platt":
            valid = math.isfinite(float(parameters["slope"])) and math.isfinite(float(parameters["intercept"]))
        else:
            valid = math.isfinite(float(parameters["temperature"])) and float(parameters["temperature"]) > 0
        return data if valid else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def calibrated_probabilities(logit, calibrator):
    """Return (AI, genuine) probabilities, or None when no valid artifact is loaded."""
    try:
        value = float(logit)
        if not math.isfinite(value) or calibrator is None:
            return None
        params = calibrator["parameters"]
        if calibrator["method"] == "platt":
            z = float(params["slope"]) * value + float(params["intercept"])
        elif calibrator["method"] == "temperature":
            z = value / float(params["temperature"])
        else:
            return None
        if not math.isfinite(z):
            return None
        ai = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        return {"ai_generated_probability": ai, "genuine_probability": 1.0 - ai}
    except (TypeError, ValueError, KeyError, OverflowError):
        return None
