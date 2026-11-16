import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.api_server as api_server


class DetectorReadinessTests(unittest.TestCase):
    def test_health_reports_detector_readiness_and_audio_unavailable_reason(self):
        reason = "PellaV2 is unavailable because the server has insufficient free memory."

        with (
            patch.dict(
                api_server.detectors,
                {"image": None, "audio": None, "video": None},
            ),
            patch.object(api_server, "audio_unavailable_reason", reason),
        ):
            response = api_server.health()

        self.assertEqual(response["status"], "offline")
        self.assertEqual(
            response["detectors"],
            {"image": False, "audio": False, "video": False},
        )
        self.assertEqual(response["audio_unavailable_reason"], reason)


if __name__ == "__main__":
    unittest.main()
