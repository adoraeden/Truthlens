import csv
import io
from pathlib import Path

import torch
from PIL import Image

from src.image_detector import ImageDetector, LOGIT_THRESHOLD

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "new_real": 0,
    "new_fake": 1,
}
OUTPUT = ROOT / "preprocessing_comparison.csv"

def resize_to_768(image):
    width, height = image.size
    scale = 768 / max(width, height)
    if scale < 1:
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.BILINEAR,
        )
    return image

def make_variants(image, detector):
    image = image.convert("RGB")

    # A: Current production preprocessing: resize + JPEG 60.
    current = detector._prepare_web_view(image)

    # B: Resize to 768px, without additional JPEG compression.
    resized = resize_to_768(image)

    # C: Resize to 768px, then JPEG 90.
    high_quality = resize_to_768(image)
    buffer = io.BytesIO()
    high_quality.save(buffer, format="JPEG", quality=90, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        high_quality = compressed.convert("RGB")

    return {
        "A_current_q60": current,
        "B_resize_only": resized,
        "C_resize_q90": high_quality,
    }

def get_logit(detector, image):
    tensor = detector.model.preprocess_input(image)
    tensor = tensor.to(detector.model.device)

    with torch.inference_mode():
        logit = detector.model.vit(tensor).reshape(-1)[0].item()

    return float(logit)

def main():
    print("Loading TruthLens model...")
    detector = ImageDetector()
    rows = []

    for folder, actual in DATASETS.items():
        directory = ROOT / folder
        files = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )

        print(f"\nEvaluating {folder}: {len(files)} images")

        for index, path in enumerate(files, 1):
            try:
                with Image.open(path) as opened:
                    opened.load()
                    original = opened.convert("RGB")

                variants = make_variants(original, detector)

                row = {
                    "filename": path.name,
                    "folder": folder,
                    "actual": actual,
                }

                for name, variant in variants.items():
                    logit = get_logit(detector, variant)
                    predicted = int(logit >= LOGIT_THRESHOLD)

                    row[f"{name}_logit"] = logit
                    row[f"{name}_predicted"] = predicted

                rows.append(row)
                print(f"  [{index}/{len(files)}] {path.name}")

            except Exception as exc:
                print(f"  ERROR: {path.name}: {exc}")

    if not rows:
        print("No images were evaluated. Check the dataset folders.")
        return

    fieldnames = list(rows[0].keys())
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== COMPARISON RESULTS ==========")
    print(f"Images evaluated: {len(rows)}")

    for name in ["A_current_q60", "B_resize_only", "C_resize_q90"]:
        tp = sum(r[f"{name}_predicted"] == 1 and r["actual"] == 1 for r in rows)
        tn = sum(r[f"{name}_predicted"] == 0 and r["actual"] == 0 for r in rows)
        fp = sum(r[f"{name}_predicted"] == 1 and r["actual"] == 0 for r in rows)
        fn = sum(r[f"{name}_predicted"] == 0 and r["actual"] == 1 for r in rows)

        accuracy = (tp + tn) / len(rows) * 100
        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    print(f"\nDetailed results saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
