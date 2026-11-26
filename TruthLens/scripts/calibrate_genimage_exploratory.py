import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrate_image_detector import (
    fit_calibrator,
    probabilities,
    metrics,
    write_reliability_svg,
)
from src.image_detector import LOGIT_THRESHOLD

DATA = ROOT / "data" / "genimage_validation_sample" / "manifest.csv"
PREDICTIONS = ROOT / "reports" / "genimage_validation_predictions.csv"
OUT = ROOT / "reports" / "genimage_calibration_exploratory"

CALIBRATION_GENERATORS = {"ADM", "BigGAN", "glide", "Midjourney"}
DEVELOPMENT_GENERATORS = {"stable_diffusion_v_1_4", "stable_diffusion_v_1_5"}
TEST_GENERATORS = {"VQDM", "wukong"}

SPLITS = {
    "calibration": CALIBRATION_GENERATORS,
    "development": DEVELOPMENT_GENERATORS,
    "test": TEST_GENERATORS,
}


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    manifest = read_csv(DATA)
    predictions = read_csv(PREDICTIONS)

    manifest_by_path = {r["archive_path"]: r for r in manifest}
    if len(manifest_by_path) != len(manifest):
        raise ValueError("Duplicate archive paths in dataset manifest.")

    split_rows = {name: [] for name in SPLITS}
    unmatched = []

    for pred in predictions:
        filename = pred["filename"]
        source = manifest_by_path.get(filename)

        if source is None:
            unmatched.append(filename)
            continue

        if pred["inference_error"].strip():
            raise ValueError(f"Inference error for {filename}")

        if pred["sha256"].lower() != source["sha256"].lower():
            raise ValueError(f"SHA-256 mismatch for {filename}")

        generator = source["generator"]
        split_name = next(
            (name for name, groups in SPLITS.items() if generator in groups),
            None,
        )
        if split_name is None:
            raise ValueError(f"Generator not assigned to a split: {generator}")

        label = pred["ground_truth"].strip().lower()
        if label not in {"ai", "nature"}:
            raise ValueError(f"Unknown ground truth {label!r} for {filename}")

        split_rows[split_name].append({
            "logit": float(pred["raw_model_logit"]),
            "truth": 1 if label == "ai" else 0,
            "sha256": source["sha256"].lower(),
            "generator_id": generator,
        })

    if unmatched:
        raise ValueError(f"{len(unmatched)} prediction filenames were missing from manifest.")

    OUT.mkdir(parents=True, exist_ok=True)

    # Verify generator separation, hash separation and minimum class counts.
    seen_generators = set()
    seen_hashes = set()
    summary = {}

    for name, rows in split_rows.items():
        generators = {r["generator_id"] for r in rows}
        hashes = [r["sha256"] for r in rows]
        truths = [r["truth"] for r in rows]

        if generators & seen_generators:
            raise ValueError(f"Generator overlap detected in {name}.")
        if seen_hashes & set(hashes) or len(hashes) != len(set(hashes)):
            raise ValueError(f"SHA-256 overlap or duplicates detected in {name}.")

        seen_generators |= generators
        seen_hashes |= set(hashes)

        genuine = truths.count(0)
        ai = truths.count(1)
        minimum = 100 if name == "calibration" else 50
        if min(genuine, ai) < minimum:
            raise ValueError(
                f"{name} has {genuine} genuine and {ai} AI; "
                f"minimum is {minimum} each."
            )

        summary[name] = {
            "count": len(rows),
            "genuine": genuine,
            "ai_generated": ai,
            "generators": sorted(generators),
        }

        # Save the exact split for review and reproducibility.
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["logit", "truth", "sha256", "generator_id"],
            )
            writer.writeheader()
            writer.writerows(rows)

    fitted = {}
    for method in ("platt", "temperature"):
        rows = split_rows["calibration"]
        fitted[method] = fit_calibrator(
            method,
            [r["logit"] for r in rows],
            [r["truth"] for r in rows],
        )

    dev = {}
    for method, params in fitted.items():
        rows = split_rows["development"]
        dev[method] = metrics(
            [r["truth"] for r in rows],
            probabilities(method, params, [r["logit"] for r in rows]),
            bins=5,
        )

    selected = min(
        fitted,
        key=lambda m: (
            dev[m]["negative_log_likelihood"],
            dev[m]["brier_score"],
        ),
    )

    test_results = {}
    for method, params in fitted.items():
        rows = split_rows["test"]
        test_results[method] = metrics(
            [r["truth"] for r in rows],
            probabilities(method, params, [r["logit"] for r in rows]),
            bins=5,
        )

    report = {
        "exploratory": True,
        "production_eligible": False,
        "dataset_version": "genimage-validation-sample-400-v1",
        "decision_threshold": LOGIT_THRESHOLD,
        "split_policy": "generator-disjoint",
        "source_lineage_verified": False,
        "original_image_lineage_verified": False,
        "counts": summary,
        "method_selection": "Lowest development NLL; Brier score tie-break",
        "selected_method": selected,
        "method": selected,
        "parameters": fitted[selected],
        "development_metrics": dev,
        "untouched_generator_test_metrics": test_results,
        "limitations": [
            "Exploratory sample only; not production validated.",
            "Source/original image lineage identifiers were not provided.",
            "Test covers VQDM and Wukong only, not all possible generators.",
            "Only 100 images are in the held-out test split.",
            "Do not interpret these estimates as universal authenticity probabilities.",
        ],
    }

    output = OUT / "calibration_candidate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_reliability_svg(OUT / "reliability.svg", test_results)

    print(json.dumps({
        "split_summary": summary,
        "selected_method": selected,
        "development": dev[selected],
        "held_out_generator_test": test_results[selected],
        "candidate_file": str(output),
        "reliability_diagram": str(OUT / "reliability.svg"),
        "production_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
