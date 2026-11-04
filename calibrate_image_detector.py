"""Fit and evaluate image-logit calibration using disjoint, source-aware CSV splits.

Each CSV must contain: logit, truth (0 genuine / 1 AI), sha256, source_id,
generator_id, original_id. The three CSVs are calibration, method-selection
development, and untouched test sets. The output is always a candidate artifact
with production_eligible=false; promotion requires a documented independent audit.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.image_detector import LOGIT_THRESHOLD
from src.image_calibration import CALIBRATOR_SCHEMA_VERSION


GROUP_COLUMNS = ("source_id", "generator_id", "original_id")


def load_split(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"logit", "truth", "sha256", *GROUP_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} requires non-empty rows and columns {sorted(required)}")
    parsed = []
    for row in rows:
        label = str(row["truth"]).strip().lower()
        if label not in {"0", "1", "real", "genuine", "fake", "ai", "ai-generated", "synthetic"}:
            raise ValueError(f"Invalid truth label {label!r} in {path}")
        truth = int(label in {"1", "fake", "ai", "ai-generated", "synthetic"})
        logit = float(row["logit"])
        if not math.isfinite(logit):
            raise ValueError(f"Non-finite logit in {path}")
        parsed.append({"truth": truth, "logit": logit, **{key: str(row[key]).strip() for key in ("sha256", *GROUP_COLUMNS)}})
    return parsed


def audit_splits(splits):
    for name, rows in splits.items():
        if any(not row[key] for row in rows for key in ("sha256", *GROUP_COLUMNS)):
            raise ValueError(f"{name} split contains missing hash/source/generator/original identifiers")
        hashes = [row["sha256"].lower() for row in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"Duplicate sha256 within {name} split")
    names = list(splits)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left, right = splits[left_name], splits[right_name]
            for key in ("sha256", *GROUP_COLUMNS):
                overlap = {row[key].lower() for row in left} & {row[key].lower() for row in right}
                if overlap:
                    raise ValueError(f"{key} overlap between {left_name} and {right_name}: {len(overlap)} identifiers")
    return {name: {"n": len(rows), "genuine": sum(r["truth"] == 0 for r in rows), "ai_generated": sum(r["truth"] == 1 for r in rows)} for name, rows in splits.items()}


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.minimum(z, 700))), np.exp(np.maximum(z, -700)) / (1 + np.exp(np.maximum(z, -700))))


def fit_calibrator(method, logits, truths):
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(truths, dtype=torch.float64)
    if torch.unique(y).numel() != 2:
        raise ValueError("Calibration split must contain both classes")
    raw = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    params = [raw, bias] if method == "platt" else [raw]
    optimizer = torch.optim.LBFGS(params, lr=0.5, max_iter=300, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        scale = torch.nn.functional.softplus(raw) if method == "platt" else torch.exp(-raw)
        z = x * scale + (bias if method == "platt" else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float((torch.nn.functional.softplus(raw) if method == "platt" else torch.exp(-raw)).detach())
    if method == "platt":
        return {"slope": scale, "intercept": float(bias.detach())}
    return {"temperature": 1 / scale}


def probabilities(method, parameters, logits):
    z = np.asarray(logits, dtype=np.float64)
    if method == "platt":
        z = parameters["slope"] * z + parameters["intercept"]
    else:
        z = z / parameters["temperature"]
    return _sigmoid(z)


def metrics(truths, probs, bins=10):
    y = np.asarray(truths, dtype=np.int64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-15, 1 - 1e-15)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece = 0.0
    reliability = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (p >= low) & ((p < high) if index < bins - 1 else (p <= high))
        count = int(mask.sum())
        if count:
            confidence, observed = float(p[mask].mean()), float(y[mask].mean())
            ece += count / len(y) * abs(confidence - observed)
            reliability.append({"bin_low": low, "bin_high": high, "count": count, "mean_predicted_ai": confidence, "observed_ai_fraction": observed})
    pos, neg = y == 1, y == 0
    roc_auc = pr_auc = None
    if pos.any() and neg.any():
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=np.float64)
        sorted_p = p[order]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_p[end] == sorted_p[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        roc_auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
        average_precision = 0.0
        previous_recall = 0.0
        for threshold in np.unique(p)[::-1]:
            selected = p >= threshold
            recall = float(y[selected].sum() / pos.sum())
            precision = float(y[selected].sum() / selected.sum())
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
        pr_auc = float(average_precision)
    by_class = {}
    for label, mask in (("genuine", neg), ("ai_generated", pos)):
        class_p = p[mask]; class_y = y[mask]
        by_class[label] = {"n": int(mask.sum()), "brier": float(np.mean((class_p - class_y) ** 2)) if mask.any() else None,
                           "negative_log_likelihood": float(-np.mean(class_y * np.log(class_p) + (1 - class_y) * np.log(1 - class_p))) if mask.any() else None,
                           "mean_predicted_ai_probability": float(class_p.mean()) if mask.any() else None}
    return {"n": len(y), "brier_score": brier, "negative_log_likelihood": nll, "expected_calibration_error_10_bins": float(ece), "roc_auc": roc_auc, "pr_auc_average_precision": pr_auc, "by_class": by_class, "reliability_bins": reliability}


def write_reliability_svg(path, metrics_by_method):
    width, height = 640, 420
    colors = {"platt": "#2563eb", "temperature": "#dc2626"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="320" y="24" text-anchor="middle" font-size="18">Reliability diagram (AI probability)</text>', '<path d="M70 350 L570 50 M70 350 L570 350 M70 350 L70 50" stroke="#555" fill="none"/>', '<text x="320" y="395" text-anchor="middle">Mean predicted AI probability</text>', '<text x="18" y="200" transform="rotate(-90 18 200)" text-anchor="middle">Observed AI fraction</text>', '<path d="M70 350 L570 50" stroke="#999" stroke-dasharray="5 5"/>']
    for method, metrics_value in metrics_by_method.items():
        points = []
        for item in metrics_value["reliability_bins"]:
            x = 70 + item["mean_predicted_ai"] * 500; y = 350 - item["observed_ai_fraction"] * 300
            points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[method]}"/>')
        lines.extend(points)
        lines.append(f'<text x="{90 if method == "platt" else 190}" y="{45 if method == "platt" else 65}" fill="{colors[method]}">{method}</text>')
    lines.append('</svg>')
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", required=True, type=Path, help="Candidate JSON path; never writes the production artifact")
    args = parser.parse_args(argv)
    splits = {"calibration": load_split(args.calibration), "development": load_split(args.development), "test": load_split(args.test)}
    counts = audit_splits(splits)
    for name, rows in splits.items():
        if min(counts[name]["genuine"], counts[name]["ai_generated"]) < 100:
            raise ValueError(f"{name} split has fewer than 100 samples in each class; refusing to claim reliable calibration")
    fitted = {method: fit_calibrator(method, [r["logit"] for r in splits["calibration"]], [r["truth"] for r in splits["calibration"]]) for method in ("platt", "temperature")}
    dev = {method: metrics([r["truth"] for r in splits["development"]], probabilities(method, fitted[method], [r["logit"] for r in splits["development"]])) for method in fitted}
    selected = min(fitted, key=lambda method: (dev[method]["negative_log_likelihood"], dev[method]["brier_score"]))
    test_results = {method: metrics([r["truth"] for r in splits["test"]], probabilities(method, fitted[method], [r["logit"] for r in splits["test"]])) for method in fitted}
    report = {"schema_version": CALIBRATOR_SCHEMA_VERSION, "production_eligible": False, "dataset_version": args.dataset_version,
              "decision_threshold": LOGIT_THRESHOLD, "calibration_split_counts": counts, "method_selection": "minimum development NLL; Brier score tie-break", "selected_method": selected,
              "method": selected, "parameters": fitted[selected], "development_metrics": dev, "untouched_test_metrics": test_results,
              "note": "Candidate only: human/source-independence review and promotion are separate. Never interpret these estimates as cross-generator performance without representative grouped test data."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_reliability_svg(args.output.with_suffix(".reliability.svg"), test_results)
    print(json.dumps({"counts": counts, "selected_method": selected, "test_metrics": test_results, "candidate_artifact": str(args.output), "production_eligible": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
