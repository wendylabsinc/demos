"""A release label alone must not hide local source changes or missing stamps."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import deployment_version as version


class DeploymentVersionTests(unittest.TestCase):
    def test_source_edits_change_digest_even_with_the_same_release_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["adapter", "g1_can_handoff", "yolo"]:
                (root / name).mkdir()
            (root / "wendy.json").write_text(
                json.dumps(dict(appId="test", version="same-label"))
            )
            (root / "build.stagefile.yaml").write_text("paths: [sample.py]\n")
            (root / "sample.py").write_text("value=1\n")
            for name in [
                "yolo/frame_limits.py",
                "yolo/confidence.py",
                "g1_can_handoff/noise_aware_planner.json",
            ]:
                (root / name).write_text("{}")
            first = version.source_version(root)
            (root / "sample.py").write_text("value=2\n")
            second = version.source_version(root)
            self.assertEqual(first["app_version"], second["app_version"])
            self.assertNotEqual(first["source_digest"], second["source_digest"])
            stamp = version.write_stamp(second, root)
            self.assertEqual(json.loads((root / version.STAMP).read_text()), stamp)
            self.assertIn("built_at_utc", stamp)

    def test_missing_stamp_is_unreported_and_reading_never_calls_git(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(version, "ROOT", Path(directory)),
            patch.object(version.subprocess, "check_output") as git,
        ):
            version.deployed_version.cache_clear()
            self.addCleanup(version.deployed_version.cache_clear)
            self.assertIsNone(version.deployed_version()["app_version"])
            git.assert_not_called()

    def test_runtime_stamp_is_cached_for_this_process(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(version, "ROOT", Path(directory)),
        ):
            version.deployed_version.cache_clear()
            self.addCleanup(version.deployed_version.cache_clear)
            path = Path(directory) / version.STAMP
            path.write_text('{"app_version":"first-build"}')
            self.assertEqual(version.deployed_version()["app_version"], "first-build")
            path.write_text('{"app_version":"later-file-change"}')
            self.assertEqual(version.deployed_version()["app_version"], "first-build")


if __name__ == "__main__":
    unittest.main()
