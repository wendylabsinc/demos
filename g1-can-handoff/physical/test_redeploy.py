"""Deployment stays behind a verified ownership check and the timing wrapper."""

from pathlib import Path
from contextlib import nullcontext
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from g1_can_handoff import redeploy


class RedeployTests(unittest.TestCase):
    def setUp(self):
        self.info = dict(
            app_version="test-version",
            git_commit="a" * 40,
            dirty=False,
            source_digest="b" * 64,
            built_at_utc="2026-10-09T00:00:00Z",
        )
        source = patch.object(redeploy, "source_version", return_value=self.info)
        stamp = patch.object(redeploy, "write_stamp", return_value=self.info)
        environment = patch.object(
            redeploy, "build_environment", return_value=nullcontext({})
        )
        source.start()
        stamp.start()
        environment.start()
        self.addCleanup(source.stop)
        self.addCleanup(stamp.stop)
        self.addCleanup(environment.stop)

    def test_failed_preflight_never_deploys_and_clears_own_marker(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(redeploy, "MARKER", Path(directory) / "lock"),
            patch("g1_can_handoff.redeploy.subprocess.run") as run,
        ):
            run.side_effect = [None, subprocess.CalledProcessError(1, ["preflight"])]
            with self.assertRaises(subprocess.CalledProcessError):
                redeploy.main([])
            self.assertEqual(run.call_count, 2)
            self.assertFalse(redeploy.MARKER.exists())

    def test_verified_deployment_uses_wrapper_and_functional_readiness(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(redeploy, "MARKER", Path(directory) / "lock"),
            patch("g1_can_handoff.redeploy.subprocess.run") as run,
        ):
            self.assertEqual(redeploy.main([]), 0)
            deployment = run.call_args_list[2]
            argv = deployment.args[0]
            self.assertEqual(argv[1], str(redeploy.WRAPPER))
            self.assertEqual(
                argv[2:],
                [
                    "--device",
                    redeploy.DEVICE,
                    "--dockerfile",
                    "build.stagefile.yaml",
                    "--yes",
                    "--detach",
                ],
            )
            self.assertIn(
                "verify_passive_deployment.py",
                deployment.kwargs["env"]["WENDY_TIMING_FUNCTIONAL_READINESS_COMMAND"],
            )
            self.assertFalse(redeploy.MARKER.exists())
            self.assertIn(redeploy.READBACK, run.call_args_list[3].args[0])

    def test_readback_failure_is_not_reported_as_deployment_success(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(redeploy, "MARKER", Path(directory) / "lock"),
            patch("g1_can_handoff.redeploy.subprocess.run") as run,
        ):
            run.side_effect = [
                None,
                None,
                None,
                subprocess.CalledProcessError(1, ["readback"]),
            ]
            with self.assertRaises(subprocess.CalledProcessError):
                redeploy.main([])
            self.assertFalse(redeploy.MARKER.exists())

    def test_check_mode_only_reads_state(self):
        with patch("g1_can_handoff.redeploy.subprocess.run") as run:
            redeploy.main(["--check"])
        run.assert_called_once()
        self.assertIn(redeploy.PREFLIGHT, run.call_args.args[0])
        self.assertNotIn("demo/start", redeploy.PREFLIGHT)


class DockerEnvironmentTests(unittest.TestCase):
    def test_missing_helper_uses_temporary_copy_without_editing_user_config(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "config.json"
            original.write_text(
                json.dumps(
                    {"credsStore": "missing-test-helper", "currentContext": "colima"}
                )
            )
            before = original.read_bytes()
            with (
                patch.dict(
                    os.environ,
                    {
                        "DOCKER_CONFIG": directory,
                        "DOCKER_HOST": "unix:///tmp/test-daemon.sock",
                    },
                    clear=True,
                ),
                patch.object(redeploy.shutil, "which", return_value=None),
            ):
                with redeploy.build_environment() as env:
                    copied = Path(env["DOCKER_CONFIG"]) / "config.json"
                    self.assertNotEqual(env["DOCKER_CONFIG"], directory)
                    self.assertNotIn("credsStore", json.loads(copied.read_text()))
                    self.assertEqual(copied.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(env["DOCKER_HOST"], "unix:///tmp/test-daemon.sock")
                self.assertFalse(copied.exists())
            self.assertEqual(original.read_bytes(), before)

    def test_available_helper_keeps_explicit_docker_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text('{"credsStore":"present"}')
            with (
                patch.dict(
                    os.environ,
                    {"DOCKER_CONFIG": directory, "DOCKER_CONTEXT": "chosen"},
                    clear=True,
                ),
                patch.object(redeploy.shutil, "which", return_value="/test/helper"),
                patch.object(redeploy.subprocess, "check_output") as inspect,
            ):
                with redeploy.build_environment() as env:
                    self.assertEqual(env["DOCKER_CONFIG"], directory)
                    self.assertEqual(env["DOCKER_CONTEXT"], "chosen")
                inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
