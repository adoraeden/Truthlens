import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.create_genimage_error_review import resolve_image_path, select_errors


def prediction(filename, truth, logit, generator="ADM", predicted=None, sha256="hash"):
    return {
        "filename": filename,
        "generator": generator,
        "ground_truth": truth,
        "predicted_label": predicted or ("ai" if logit >= 2.0 else "nature"),
        "raw_model_logit": str(logit),
        "decision_threshold": "2.0",
        "sha256": sha256,
        "inference_error": "",
    }


class GenImageErrorReviewTests(unittest.TestCase):
    def test_error_selection_is_raw_logit_based_and_nearest_boundary_first(self):
        rows = [
            prediction("fn_far", "ai", -1.0),
            prediction("fn_near", "ai", 1.9),
            prediction("fp_near", "nature", 2.01),
            prediction("fp_far", "nature", 4.0),
            prediction("correct", "nature", 1.0),
            {**prediction("bad", "ai", 1.0), "inference_error": "decode failed"},
        ]
        false_negatives, false_positives = select_errors(rows, 2.0)
        self.assertEqual([row["filename"] for row in false_negatives], ["fn_near", "fn_far"])
        self.assertEqual([row["filename"] for row in false_positives], ["fp_near", "fp_far"])
        self.assertAlmostEqual(false_negatives[0]["distance_from_threshold"], 0.1)

    def test_manifest_path_resolution_checks_metadata_and_content_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "ADM" / "ai" / "sample.png"
            image_path.parent.mkdir(parents=True)
            payload = b"sample bytes"
            image_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            row = prediction("ADM/ai/sample.png", "ai", 1.0, sha256=digest)
            manifest = {
                "ADM/ai/sample.png": {
                    "archive_path": "ADM/ai/sample.png", "generator": "ADM", "label": "ai", "sha256": digest,
                }
            }
            resolved, error = resolve_image_path(row, root, manifest)
            self.assertEqual(resolved, image_path.resolve())
            self.assertEqual(error, "")

            missing_row = prediction("ADM/ai/missing.png", "ai", 1.0, sha256=digest)
            resolved_missing, missing_error = resolve_image_path(missing_row, root, manifest)
            self.assertIsNone(resolved_missing)
            self.assertIn("No matching archive_path", missing_error)

            mismatch_manifest = {"ADM/ai/sample.png": {**manifest["ADM/ai/sample.png"], "sha256": "0" * 64}}
            resolved_mismatch, mismatch_error = resolve_image_path(row, root, mismatch_manifest)
            self.assertEqual(resolved_mismatch, image_path.resolve())
            self.assertIn("SHA-256", mismatch_error)


if __name__ == "__main__":
    unittest.main()
