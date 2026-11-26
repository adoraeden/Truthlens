import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_genimage import calculate_metrics, validate_manifest


class GenImageEvaluationTests(unittest.TestCase):
    def test_metrics_use_ai_as_positive_class_and_report_expected_counts(self):
        rows = [
            {"ground_truth": "ai", "predicted_label": "ai"},
            {"ground_truth": "ai", "predicted_label": "nature"},
            {"ground_truth": "nature", "predicted_label": "nature"},
            {"ground_truth": "nature", "predicted_label": "ai"},
            {"ground_truth": "ai", "predicted_label": "", "inference_error": "decode failed"},
        ]
        result = calculate_metrics(rows)
        self.assertEqual(result["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(result["sample_counts"]["total"], 5)
        self.assertEqual(result["sample_counts"]["scored"], 4)
        self.assertEqual(result["sample_counts"]["inference_errors"], 1)
        self.assertEqual(result["sample_counts"]["classes_total"], {"ai": 3, "nature": 2})
        self.assertEqual(result["sample_counts"]["classes_scored"], {"ai": 2, "nature": 2})
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["metrics"]["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["metrics"]["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["metrics"]["false_negative_rate"], 0.5)

    def test_manifest_validation_verifies_hashes_and_exact_class_sample_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for label, filename, content in (("ai", "fake.png", b"ai sample"), ("nature", "real.png", b"real sample")):
                relative = Path("Toy") / label / filename
                image_path = root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(content)
                rows.append({
                    "archive_path": relative.as_posix(), "generator": "Toy", "label": label,
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            samples = validate_manifest(root, expected_per_class=1, generators=("Toy",))
            self.assertEqual(len(samples), 2)
            self.assertEqual({sample["ground_truth"] for sample in samples}, {"ai", "nature"})

            rows[0]["sha256"] = "0" * 64
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_manifest(root, expected_per_class=1, generators=("Toy",))

    def test_manifest_rejects_missing_class_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "Toy" / "ai" / "one.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"ai")
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("archive_path", "generator", "label", "sha256"))
                writer.writeheader()
                writer.writerow({"archive_path": "Toy/ai/one.png", "generator": "Toy", "label": "ai", "sha256": hashlib.sha256(b"ai").hexdigest()})
            with self.assertRaisesRegex(ValueError, "Expected 1 images per generator/class"):
                validate_manifest(root, expected_per_class=1, generators=("Toy",))


if __name__ == "__main__":
    unittest.main()
