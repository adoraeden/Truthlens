import unittest
from unittest.mock import patch

import src.api_server as api_server


class PartialDetectorReadinessTests(unittest.TestCase):
    def test_health_reports_partial_readiness_without_marking_service_offline(self):
        unavailable_reason = "PellaV2 is unavailable because of the RAM limit."

        with (
            patch.dict(
                api_server.detectors,
                {
                    "image": object(),
                    "audio": None,
                    "video": object(),
                },
            ),
            patch.object(
                api_server,
                "audio_unavailable_reason",
                unavailable_reason,
            ),
        ):
            response = api_server.health()

        self.assertEqual(response["status"], "online")
        self.assertEqual(
            response["detectors"],
            {
                "image": True,
                "audio": False,
                "video": True,
            },
        )
        self.assertEqual(response["audio_unavailable_reason"], unavailable_reason)


if __name__ == "__main__":
    unittest.main()
