import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_genimage_thresholds import (
    EXPECTED_GENERATORS,
    build_thresholds,
    calculate_generator_metrics,
    calculate_metrics,
    current_threshold_errors,
    load_predictions,
)


def row(filename, generator, truth, logit, threshold=2.0, error=""):
    return {
        "filename": filename,
        "generator": generator,
        "ground_truth": truth,
        "raw_model_logit": logit,
        "decision_threshold": threshold,
        "inference_error": error,
    }


class GenImageThresholdTests(unittest.TestCase):
    def test_metrics_and_counts_use_ai_as_positive_and_boundary_is_inclusive(self):
        rows = [
            row("tp", "ADM", "ai", 2.0),
            row("fn", "ADM", "ai", 1.9),
            row("tn", "ADM", "nature", 1.0),
            row("fp", "ADM", "nature", 2.2),
            row("error", "ADM", "ai", None, error="bad image"),
        ]
        result = calculate_metrics(rows, 2.0)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual((result["total_count"], result["scored_count"], result["inference_error_count"]), (5, 4, 1))
        self.assertEqual((result["ai_count"], result["nature_count"]), (3, 2))
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["specificity"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["fpr"], 0.5)
        self.assertAlmostEqual(result["fnr"], 0.5)

    def test_threshold_grid_spans_observed_range_and_always_includes_production_cutoff(self):
        rows = [row("a", "ADM", "ai", -0.12), row("b", "ADM", "nature", 3.07)]
        thresholds = build_thresholds(rows, 2.0)
        self.assertEqual(thresholds, [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25])
        self.assertTrue(all(abs((right - left) / 0.25 - round((right - left) / 0.25)) < 1e-9 for left, right in zip(thresholds, thresholds[1:])))

    def test_false_negatives_are_sorted_closest_to_boundary_first(self):
        rows = [
            row("far", "ADM", "ai", -1.0),
            row("near", "ADM", "ai", 1.9),
            row("fp", "ADM", "nature", 2.1),
        ]
        false_negatives, false_positives = current_threshold_errors(rows, 2.0)
        self.assertEqual([item["filename"] for item in false_negatives], ["near", "far"])
        self.assertEqual([item["filename"] for item in false_positives], ["fp"])

    def test_generator_rates_return_all_generators_and_class_counts(self):
        rows = [row("a", "ADM", "ai", 3.0), row("n", "ADM", "nature", 3.0)]
        results = calculate_generator_metrics(rows, 2.0)
        self.assertEqual([item["generator"] for item in results], list(EXPECTED_GENERATORS))
        adm = results[0]
        self.assertEqual((adm["ai_count"], adm["nature_count"], adm["tp"], adm["fp"]), (1, 1, 1, 1))
        self.assertEqual(adm["recall"], 1.0)
        self.assertEqual(adm["fpr"], 1.0)
        self.assertIsNone(results[1]["recall"])
        self.assertIsNone(results[1]["fpr"])

    def test_loads_actual_prediction_schema_without_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.csv"
            fields = ["filename", "generator", "ground_truth", "predicted_label", "raw_model_logit", "decision_threshold", "inference_error"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"filename": "sample.png", "generator": "ADM", "ground_truth": "ai", "predicted_label": "nature", "raw_model_logit": "1.5", "decision_threshold": "2", "inference_error": ""})
            parsed, threshold = load_predictions(path, expected_rows=1)
            self.assertEqual(threshold, 2.0)
            self.assertEqual(parsed[0]["raw_model_logit"], 1.5)
