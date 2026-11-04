r"""Sweep raw-logit thresholds over saved GenImage predictions (no model inference).

Run from the repository root with:
    .\.venv\Scripts\python.exe scripts\analyze_genimage_thresholds.py

The sweep is exploratory on the same validation sample and does not set a
production threshold.
"""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "genimage_validation_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports"
EXPECTED_GENERATORS = (
    "ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5", "VQDM", "wukong",
)
POSITIVE = "ai"
NEGATIVE = "nature"
THRESHOLD_STEP = 0.25


def load_predictions(path=DEFAULT_INPUT, expected_rows=400):
    """Read the prediction ledger's raw logits, labels, threshold, and errors."""
    required = {"filename", "generator", "ground_truth", "raw_model_logit", "decision_threshold", "inference_error"}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Prediction CSV is missing required columns: {missing}")
        rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} prediction rows, found {len(rows)}")
    if not rows:
        raise ValueError("Prediction CSV contains no rows")

    parsed = []
    filenames = set()
    thresholds = set()
    for line, row in enumerate(rows, start=2):
        filename = row["filename"].strip()
        generator = row["generator"].strip()
        truth = row["ground_truth"].strip().lower()
        if not filename or filename in filenames:
            raise ValueError(f"Missing or duplicate filename on CSV row {line}: {filename!r}")
        filenames.add(filename)
        if generator not in EXPECTED_GENERATORS:
            raise ValueError(f"Unknown generator on CSV row {line}: {generator!r}")
        if truth not in {POSITIVE, NEGATIVE}:
            raise ValueError(f"Unknown ground-truth label on CSV row {line}: {truth!r}")
        try:
            threshold = float(row["decision_threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid decision threshold on CSV row {line}") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Non-finite decision threshold on CSV row {line}")
        thresholds.add(threshold)
        error = row["inference_error"].strip()
        raw_logit = row["raw_model_logit"].strip()
        if error:
            logit = None
        else:
            try:
                logit = float(raw_logit)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid logit without an inference error on CSV row {line}") from exc
            if not math.isfinite(logit):
                raise ValueError(f"Non-finite logit on CSV row {line}")
        parsed.append({
            "filename": filename,
            "generator": generator,
            "ground_truth": truth,
            "raw_model_logit": logit,
            "decision_threshold": threshold,
            "inference_error": error,
        })
    if len(thresholds) != 1:
        raise ValueError(f"Prediction rows contain inconsistent decision thresholds: {sorted(thresholds)}")
    return parsed, thresholds.pop()


def calculate_metrics(rows, threshold):
    """Compute AI-positive binary metrics; failed inferences remain in coverage counts."""
    total_ai = sum(row["ground_truth"] == POSITIVE for row in rows)
    total_nature = sum(row["ground_truth"] == NEGATIVE for row in rows)
    scored = [row for row in rows if row["raw_model_logit"] is not None and not row["inference_error"]]
    scored_ai = sum(row["ground_truth"] == POSITIVE for row in scored)
    scored_nature = sum(row["ground_truth"] == NEGATIVE for row in scored)
    tp = fp = tn = fn = 0
    for row in scored:
        actual_ai = row["ground_truth"] == POSITIVE
        predicted_ai = row["raw_model_logit"] >= threshold
        if actual_ai and predicted_ai:
            tp += 1
        elif actual_ai:
            fn += 1
        elif predicted_ai:
            fp += 1
        else:
            tn += 1

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "threshold": threshold,
        "total_count": len(rows),
        "scored_count": len(scored),
        "inference_error_count": len(rows) - len(scored),
        "ai_count": total_ai,
        "nature_count": total_nature,
        "scored_ai_count": scored_ai,
        "scored_nature_count": scored_nature,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(scored)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
    }


def build_thresholds(rows, current_threshold, step=THRESHOLD_STEP):
    logits = [row["raw_model_logit"] for row in rows if row["raw_model_logit"] is not None]
    if not logits:
        raise ValueError("No valid raw logits are available for the threshold sweep")
    if step <= 0 or not math.isfinite(step):
        raise ValueError("Threshold step must be a positive finite number")
    low_index = math.floor(min(logits) / step)
    high_index = math.ceil(max(logits) / step)
    thresholds = {round(index * step, 10) for index in range(low_index, high_index + 1)}
    thresholds.add(round(current_threshold, 10))
    return sorted(thresholds)


def calculate_generator_metrics(rows, threshold, generators=EXPECTED_GENERATORS):
    results = []
    for generator in generators:
        subset = [row for row in rows if row["generator"] == generator]
        metrics = calculate_metrics(subset, threshold)
        results.append({
            "threshold": threshold,
            "generator": generator,
            "ai_count": metrics["ai_count"],
            "nature_count": metrics["nature_count"],
            "scored_ai_count": metrics["scored_ai_count"],
            "scored_nature_count": metrics["scored_nature_count"],
            "inference_error_count": metrics["inference_error_count"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
        })
    return results


def current_threshold_errors(rows, threshold):
    false_negatives = [
        row for row in rows
        if not row["inference_error"] and row["raw_model_logit"] is not None
        and row["ground_truth"] == POSITIVE and row["raw_model_logit"] < threshold
    ]
    false_positives = [
        row for row in rows
        if not row["inference_error"] and row["raw_model_logit"] is not None
        and row["ground_truth"] == NEGATIVE and row["raw_model_logit"] >= threshold
    ]
    false_negatives.sort(key=lambda row: (threshold - row["raw_model_logit"], row["filename"]))
    false_positives.sort(key=lambda row: (row["raw_model_logit"] - threshold, row["filename"]))
    return false_negatives, false_positives


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(path, predictions, current_threshold, overall, by_generator, false_negatives, false_positives):
    current = next(row for row in overall if math.isclose(row["threshold"], current_threshold, abs_tol=1e-9))
    lines = [
        "# GenImage Exploratory Threshold Analysis", "",
        f"- Source: `{DEFAULT_INPUT.name}`; {len(predictions)} saved prediction rows. No model inference was run.",
        f"- Ground truth: `{POSITIVE}` is positive; `{NEGATIVE}` is negative.",
        f"- Current production raw-logit threshold: **{current_threshold:g}** (included in the sweep).",
        "- This sweep is exploratory on the same evaluation sample. It does not independently validate any threshold and does not change production behavior.",
        "- `threshold_relative_score_uncalibrated` is not a probability; the sweep uses raw logits only.", "",
        "## Current-threshold metrics", "",
        f"Scored {current['scored_count']}/{current['total_count']} rows; inference errors: {current['inference_error_count']}.",
        f"Class counts: AI={current['ai_count']} (scored {current['scored_ai_count']}), nature={current['nature_count']} (scored {current['scored_nature_count']}).",
        f"TP={current['tp']}, FP={current['fp']}, TN={current['tn']}, FN={current['fn']}.",
        f"Accuracy={_fmt(current['accuracy'])}; precision={_fmt(current['precision'])}; recall={_fmt(current['recall'])}; specificity={_fmt(current['specificity'])}; F1={_fmt(current['f1'])}; balanced accuracy={_fmt(current['balanced_accuracy'])}; FPR={_fmt(current['fpr'])}; FNR={_fmt(current['fnr'])}.",
        "", "## Generator recall and false-positive rate at the current threshold", "",
        "| Generator | AI count | Nature count | Recall | FPR | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in by_generator:
        if math.isclose(item["threshold"], current_threshold, abs_tol=1e-9):
            lines.append(f"| {item['generator']} | {item['ai_count']} | {item['nature_count']} | {_fmt(item['recall'])} | {_fmt(item['fpr'])} | {item['tp']} | {item['fp']} | {item['tn']} | {item['fn']} |")
    lines.extend(["", "## Current-threshold false negatives", "", "Sorted by raw logit nearest to the boundary first."])
    if false_negatives:
        lines.extend(f"- `{row['filename']}` ({row['generator']}): logit {row['raw_model_logit']:.6g}; {current_threshold - row['raw_model_logit']:.6g} below threshold" for row in false_negatives)
    else:
        lines.append("- None.")
    lines.extend(["", "## Current-threshold false positives", "", "Sorted by raw logit nearest to the boundary first."])
    if false_positives:
        lines.extend(f"- `{row['filename']}` ({row['generator']}): logit {row['raw_model_logit']:.6g}; {row['raw_model_logit'] - current_threshold:.6g} above threshold" for row in false_positives)
    else:
        lines.append("- None.")
    neighbor = next((row for row in overall if math.isclose(row["threshold"], current_threshold - THRESHOLD_STEP, abs_tol=1e-9)), None)
    lines.extend(["", "## Recall / false-positive trade-off", "", "Lowering the threshold can classify more AI samples as positive (higher recall), while also classifying more nature samples as positive (higher FPR). Raising it generally moves in the opposite direction. These are empirical trade-offs on this sample, not a basis for automatically changing production."])
    if neighbor:
        lines.append(f"At the adjacent lower threshold {neighbor['threshold']:g}, recall is {_fmt(neighbor['recall'])} and FPR is {_fmt(neighbor['fpr'])}; at the current threshold {current_threshold:g}, recall is {_fmt(current['recall'])} and FPR is {_fmt(current['fpr'])}.")
    lines.extend(["", "No threshold is selected or applied by this analysis.", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def analyze(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR):
    predictions, current_threshold = load_predictions(input_path)
    thresholds = build_thresholds(predictions, current_threshold)
    overall = [calculate_metrics(predictions, threshold) for threshold in thresholds]
    by_generator = [
        row
        for threshold in thresholds
        for row in calculate_generator_metrics(predictions, threshold)
    ]
    false_negatives, false_positives = current_threshold_errors(predictions, current_threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_fields = list(overall[0])
    generator_fields = list(by_generator[0])
    write_csv(output_dir / "genimage_threshold_sweep_overall.csv", overall, overall_fields)
    write_csv(output_dir / "genimage_threshold_sweep_by_generator.csv", by_generator, generator_fields)
    write_markdown(
        output_dir / "genimage_threshold_analysis.md", predictions, current_threshold,
        overall, by_generator, false_negatives, false_positives,
    )
    return {
        "current_threshold": current_threshold,
        "overall": overall,
        "by_generator": by_generator,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "outputs": output_dir,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = analyze(args.input, args.output_dir)
    current = next(row for row in result["overall"] if math.isclose(row["threshold"], result["current_threshold"], abs_tol=1e-9))
    print(f"Loaded {sum(row['total_count'] for row in result['overall'][:1])} saved image rows; no inference run.")
    print(f"Current threshold {result['current_threshold']:g}: TP={current['tp']} FP={current['fp']} TN={current['tn']} FN={current['fn']} recall={_fmt(current['recall'])} FPR={_fmt(current['fpr'])}.")
    print(f"Current-threshold false negatives: {len(result['false_negatives'])}; false positives: {len(result['false_positives'])}.")
    print(f"Wrote threshold sweep and report to {result['outputs']}.")
    print("Exploratory sample only; no production threshold was selected or changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
