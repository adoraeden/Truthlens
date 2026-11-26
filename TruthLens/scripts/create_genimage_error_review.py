"""Create labeled contact sheets for saved GenImage threshold errors; no inference."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "reports" / "genimage_validation_predictions.csv"
DEFAULT_MANIFEST = ROOT / "data" / "genimage_validation_sample" / "manifest.csv"
DEFAULT_DATASET = ROOT / "data" / "genimage_validation_sample"
DEFAULT_OUTPUT = ROOT / "reports" / "genimage_error_review"
CURRENT_THRESHOLD = 2.0
SHEET_COLUMNS = 4
CARD_WIDTH = 390
IMAGE_WIDTH = 350
IMAGE_HEIGHT = 270
CARD_HEIGHT = 430
OUTER_PADDING = 24
GAP = 16


def read_csv(path, required_columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required_columns).issubset(reader.fieldnames):
            missing = sorted(set(required_columns) - set(reader.fieldnames or []))
            raise ValueError(f"{path} is missing required columns: {missing}")
        return list(reader)


def read_manifest(path=DEFAULT_MANIFEST):
    rows = read_csv(path, ("archive_path", "generator", "label", "sha256"))
    index = {}
    for row in rows:
        key = row["archive_path"].replace("\\", "/")
        if key in index:
            raise ValueError(f"Duplicate archive_path in evaluation manifest: {key}")
        index[key] = row
    return index


def select_errors(rows, threshold=CURRENT_THRESHOLD):
    """Select false negatives/positives from labels and raw logits, not saved verdicts."""
    if not math.isfinite(threshold):
        raise ValueError("Decision threshold must be finite")
    false_negatives, false_positives = [], []
    for row in rows:
        if row.get("inference_error", "").strip():
            continue
        truth = row["ground_truth"].strip().lower()
        if truth not in {"ai", "nature"}:
            raise ValueError(f"Unknown ground truth label: {truth!r}")
        logit = float(row["raw_model_logit"])
        row_threshold = float(row["decision_threshold"])
        if not math.isfinite(logit) or not math.isclose(row_threshold, threshold, abs_tol=1e-9):
            raise ValueError(f"Invalid logit or row threshold for {row.get('filename', 'sample')}")
        predicted = "ai" if logit >= threshold else "nature"
        stored_prediction = row.get("predicted_label", "").strip().lower()
        if stored_prediction and stored_prediction != predicted:
            raise ValueError(f"Saved prediction disagrees with raw logit for {row.get('filename', 'sample')}")
        item = {**row, "raw_model_logit": logit, "distance_from_threshold": abs(logit - threshold), "predicted_label": predicted}
        if truth == "ai" and predicted == "nature":
            false_negatives.append(item)
        elif truth == "nature" and predicted == "ai":
            false_positives.append(item)
    false_negatives.sort(key=lambda row: (row["distance_from_threshold"], row["filename"]))
    false_positives.sort(key=lambda row: (row["distance_from_threshold"], row["filename"]))
    return false_negatives, false_positives


def resolve_image_path(row, dataset_root=DEFAULT_DATASET, manifest_index=None):
    """Resolve a prediction filename through the manifest and verify its hash."""
    if manifest_index is None:
        manifest_index = read_manifest()
    dataset_root = Path(dataset_root).resolve()
    key = row["filename"].replace("\\", "/")
    manifest_row = manifest_index.get(key)
    if manifest_row is None:
        return None, "No matching archive_path in evaluation manifest"
    if manifest_row["generator"] != row["generator"] or manifest_row["label"].strip().lower() != row["ground_truth"].strip().lower():
        return None, "Prediction metadata does not match evaluation manifest"
    path = (dataset_root / key).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError:
        return None, "Manifest image path escapes dataset directory"
    if not path.is_file():
        return path, "Image file is missing"
    expected = manifest_row["sha256"].strip().lower()
    observed = str(row.get("sha256", "")).strip().lower()
    if expected and observed and expected != observed:
        return path, "Prediction and manifest SHA-256 values differ"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected and actual != expected:
        return path, "Extracted image SHA-256 differs from evaluation manifest"
    return path, ""


def _font(size, bold=False):
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _load_review_image(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        return image.copy()


def create_contact_sheet(title, rows, dataset_root, manifest_index, output_path):
    rows = list(rows)
    columns = min(SHEET_COLUMNS, max(1, len(rows)))
    count_rows = max(1, math.ceil(len(rows) / columns))
    title_height = 64
    width = OUTER_PADDING * 2 + columns * CARD_WIDTH + (columns - 1) * GAP
    height = title_height + OUTER_PADDING + count_rows * CARD_HEIGHT + (count_rows - 1) * GAP + OUTER_PADDING
    canvas = Image.new("RGB", (width, height), (13, 17, 30))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = _font(30, True), _font(19, True), _font(16)
    draw.text((OUTER_PADDING, 17), title, font=title_font, fill=(236, 243, 255))
    if not rows:
        draw.text((OUTER_PADDING, title_height + 20), "No samples in this error category.", font=label_font, fill=(183, 194, 216))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return []

    resolved_rows = []
    start_y = title_height
    for index, row in enumerate(rows):
        col, line = index % columns, index // columns
        x = OUTER_PADDING + col * (CARD_WIDTH + GAP)
        y = start_y + line * (CARD_HEIGHT + GAP)
        draw.rounded_rectangle((x, y, x + CARD_WIDTH, y + CARD_HEIGHT), radius=12, fill=(25, 31, 49), outline=(70, 84, 115), width=2)
        image_box = (x + 20, y + 18, x + 20 + IMAGE_WIDTH, y + 18 + IMAGE_HEIGHT)
        image_path, resolution_error = resolve_image_path(row, dataset_root, manifest_index)
        if image_path is not None and not resolution_error:
            try:
                sample_image = _load_review_image(image_path)
                sample_image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                px = image_box[0] + (IMAGE_WIDTH - sample_image.width) // 2
                py = image_box[1] + (IMAGE_HEIGHT - sample_image.height) // 2
                draw.rectangle(image_box, fill=(8, 11, 20))
                canvas.paste(sample_image, (px, py))
            except (OSError, ValueError) as exc:
                resolution_error = f"Image decode failed: {exc}"
        else:
            draw.rectangle(image_box, fill=(8, 11, 20))
        if resolution_error:
            lines = _wrap_text(draw, "IMAGE UNAVAILABLE · " + resolution_error, label_font, IMAGE_WIDTH - 16)
            cursor_y = image_box[1] + 10
            for text in lines[:5]:
                draw.text((image_box[0] + 8, cursor_y), text, font=label_font, fill=(255, 151, 151))
                cursor_y += 26

        text_y = y + 304
        details = [
            f"Generator: {row['generator']}",
            f"Ground truth: {row['ground_truth'].upper()}",
            f"Predicted: {row['predicted_label'].upper()}",
            f"Raw logit: {row['raw_model_logit']:.6f}",
            f"Distance from threshold {CURRENT_THRESHOLD:g}: {row['distance_from_threshold']:.6f} {'below' if row['raw_model_logit'] < CURRENT_THRESHOLD else 'above'}",
        ]
        for detail_index, text in enumerate(details):
            font = label_font if detail_index == 0 else small_font
            color = (241, 210, 110) if detail_index == 0 else (209, 219, 239)
            wrapped = _wrap_text(draw, text, font, CARD_WIDTH - 32)
            for wrapped_line in wrapped:
                draw.text((x + 16, text_y), wrapped_line, font=font, fill=color)
                text_y += 23
        resolved_rows.append((row, image_path, resolution_error))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return resolved_rows


def write_error_details(path, false_negatives, false_positives, dataset_root, manifest_index):
    fields = [
        "error_type", "filename", "image_path", "generator", "ground_truth", "predicted_label",
        "raw_model_logit", "decision_threshold", "distance_from_threshold", "sha256", "image_review_status",
    ]
    output = []
    for error_type, rows in (("false_negative", false_negatives), ("false_positive", false_positives)):
        for row in rows:
            image_path, error = resolve_image_path(row, dataset_root, manifest_index)
            status = error or "available"
            if not error and image_path is not None:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    status = f"Image decode failed: {exc}"
            output.append({
                "error_type": error_type,
                "filename": row["filename"],
                "image_path": str(image_path) if image_path else "",
                "generator": row["generator"],
                "ground_truth": row["ground_truth"],
                "predicted_label": row["predicted_label"],
                "raw_model_logit": f"{row['raw_model_logit']:.10g}",
                "decision_threshold": f"{CURRENT_THRESHOLD:g}",
                "distance_from_threshold": f"{row['distance_from_threshold']:.10g}",
                "sha256": row.get("sha256", ""),
                "image_review_status": status,
            })
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return output


def create_review(predictions_path=DEFAULT_PREDICTIONS, manifest_path=DEFAULT_MANIFEST, dataset_root=DEFAULT_DATASET, output_dir=DEFAULT_OUTPUT):
    predictions = read_csv(predictions_path, ("filename", "generator", "ground_truth", "predicted_label", "raw_model_logit", "decision_threshold", "sha256", "inference_error"))
    manifest_index = read_manifest(manifest_path)
    false_negatives, false_positives = select_errors(predictions, CURRENT_THRESHOLD)
    output_dir = Path(output_dir)
    create_contact_sheet("FALSE NEGATIVES · AI classified as nature", false_negatives, dataset_root, manifest_index, output_dir / "false_negatives.png")
    create_contact_sheet("FALSE POSITIVES · nature classified as AI", false_positives, dataset_root, manifest_index, output_dir / "false_positives.png")
    errors = write_error_details(output_dir / "error_details.csv", false_negatives, false_positives, dataset_root, manifest_index)
    missing = [row for row in errors if row["image_review_status"] != "available"]
    return {
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "missing_or_unreadable_count": len(missing),
        "review_errors": missing,
        "outputs": output_dir,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = create_review(args.predictions, args.manifest, args.dataset, args.output_dir)
    print(f"Selected {result['false_negative_count']} false negatives and {result['false_positive_count']} false positives at threshold {CURRENT_THRESHOLD:g}.")
    print(f"Missing/unreadable images: {result['missing_or_unreadable_count']}.")
    for error in result["review_errors"]:
        print(f"  {error['filename']}: {error['image_review_status']}")
    print(f"Contact sheets and details saved to {result['outputs']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
