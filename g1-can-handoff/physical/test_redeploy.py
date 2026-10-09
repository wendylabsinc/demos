"""Deployment stays behind a verified ownership check and the timing wrapper."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from g1_can_handoff import redeploy


class RedeployTests(unittest.TestCase):
    def test_failed_preflight_never_deploys_and_clears_own_marker(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(redeploy,'MARKER',Path(directory)/'lock'),\
             patch('g1_can_handoff.redeploy.subprocess.run') as run:
            run.side_effect=[None,subprocess.CalledProcessError(1,['preflight'])]
            with self.assertRaises(subprocess.CalledProcessError):redeploy.main([])
            self.assertEqual(run.call_count,2)
            self.assertFalse(redeploy.MARKER.exists())

    def test_verified_deployment_uses_wrapper_and_functional_readiness(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(redeploy,'MARKER',Path(directory)/'lock'),\
             patch('g1_can_handoff.redeploy.subprocess.run') as run:
            self.assertEqual(redeploy.main([]),0)
            argv=run.call_args.args[0]
            self.assertEqual(argv[1],str(redeploy.WRAPPER))
            self.assertEqual(argv[2:],['--device',redeploy.DEVICE,'--dockerfile',
                                    'build.stagefile.yaml','--yes','--detach'])
            self.assertIn('verify_passive_deployment.py',run.call_args.kwargs['env']['WENDY_TIMING_FUNCTIONAL_READINESS_COMMAND'])
            self.assertFalse(redeploy.MARKER.exists())

    def test_check_mode_only_reads_state(self):
        with patch('g1_can_handoff.redeploy.subprocess.run') as run:
            redeploy.main(['--check'])
        run.assert_called_once()
        self.assertIn(redeploy.PREFLIGHT,run.call_args.args[0])
        self.assertNotIn('demo/start',redeploy.PREFLIGHT)


if __name__=='__main__':unittest.main()
