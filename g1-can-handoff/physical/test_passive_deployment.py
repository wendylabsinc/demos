"""Readiness must tolerate a bounded replacement/startup gap without motion."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import verify_passive_deployment


class PassiveDeploymentTests(unittest.TestCase):
    def client(self):return SimpleNamespace(tool=Mock(),process=SimpleNamespace(terminate=Mock()))

    def test_startup_transport_gap_retries_then_accepts_the_verified_read(self):
        client=self.client()
        client.tool.side_effect=[RuntimeError('Container not yet running'),
            {'structuredContent':{'exit_code':0,'stdout':'{"startup_verified":true}\n'}}]
        with patch('verify_passive_deployment.Wendy',return_value=client),\
             patch('verify_passive_deployment.time.sleep') as sleep,redirect_stdout(io.StringIO()):
            self.assertEqual(verify_passive_deployment.main(),0)
        self.assertEqual(client.tool.call_count,2);sleep.assert_called_once_with(1)
        client.process.terminate.assert_called_once()
        for call in client.tool.call_args_list:
            self.assertEqual(call.args[0],'container_exec')
            self.assertEqual(call.args[1]['app_name'],'DEV-g1-handoff-physical')
            # Verify the policy on the deployed status, as well as source
            # hashes; a correct build with the old environment is insufficient.
            script=call.args[1]['command'][2]
            self.assertIn("status['perception_policy']=='phase_boundary'",script)
            self.assertIn("status['periodic_yolo_recheck_s']",script)

    def test_persistent_transport_failure_stops_at_the_attempt_limit(self):
        client=self.client();client.tool.side_effect=RuntimeError('Container unavailable')
        with patch('verify_passive_deployment.Wendy',return_value=client),\
             patch('verify_passive_deployment.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'Container unavailable'):verify_passive_deployment.main()
        self.assertEqual(client.tool.call_count,8)
        client.process.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
