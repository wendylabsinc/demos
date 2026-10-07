"""Uncertain command acceptance must cancel and retain evidence, never retry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import record_demo_attempt as runner


class AttemptTests(unittest.TestCase):
    def snapshot(self):
        return {'status':{'demo_running':False,'adapter_initialized':False,'phase':'PASSIVE',
                          'motion_commands_sent':0,'demo_result':None},'events':[]}

    def test_start_is_never_repeated_after_accepted_command_transport_failure(self):
        client=Mock()
        client.request.side_effect=[{'demo_running':True},RuntimeError('transport lost'),{'phase':'FROZEN'}]
        with tempfile.TemporaryDirectory() as directory,\
             patch('sys.argv',['runner','--start','--output',directory]),\
             patch.object(runner,'Wendy',return_value=client),\
             patch.object(runner,'snapshot',return_value=self.snapshot()):
            with self.assertRaises(SystemExit):runner.main()
            evidence=json.loads(next(Path(directory).glob('20*.json')).read_text())
            self.assertIn('transport lost',evidence['runner_error'])
            self.assertIn('error_stop',evidence)
        self.assertEqual(client.request.call_args_list[0].kwargs,{'action':'start'})
        self.assertEqual(client.request.call_args_list[-1].kwargs,{'action':'stop'})
        self.assertEqual(sum(call.kwargs.get('action')=='start' for call in client.request.call_args_list),1)
        client.process.terminate.assert_called_once()

    def test_explicit_pregrasp_retry_uses_controller_admission(self):
        client=Mock();client.request.return_value={'demo_running':False}
        snapshot=self.snapshot();snapshot['status'].update(adapter_initialized=True,phase='FROZEN')
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['runner','--start','--resume-pregrasp','--output',directory]),\
             patch.object(runner,'Wendy',return_value=client),patch.object(runner,'snapshot',return_value=snapshot):
            with self.assertRaises(SystemExit):runner.main()
        self.assertEqual(client.request.call_args_list[0].kwargs,{'action':'start'})
        self.assertEqual(sum(c.kwargs.get('action')=='start' for c in client.request.call_args_list),1)

    def test_retry_flag_never_starts_from_holding_can(self):
        client=Mock();snapshot=self.snapshot()
        snapshot['status'].update(adapter_initialized=True,phase='HOLDING_CAN')
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['runner','--start','--resume-pregrasp','--output',directory]),\
             patch.object(runner,'Wendy',return_value=client),patch.object(runner,'snapshot',return_value=snapshot):
            with self.assertRaises(SystemExit):runner.main()
        client.request.assert_not_called()

    def test_capture_without_start_never_requests_motion(self):
        client=Mock()
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['runner','--output',directory]),\
             patch.object(runner,'Wendy',return_value=client),patch.object(runner,'snapshot',return_value=self.snapshot()):
            runner.main()
        client.request.assert_not_called()

    def test_lost_start_response_also_requests_stop_without_replaying_start(self):
        client=Mock();client.request.side_effect=[RuntimeError('start response lost'),{'phase':'FROZEN'}]
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['runner','--start','--output',directory]),\
             patch.object(runner,'Wendy',return_value=client),patch.object(runner,'snapshot',return_value=self.snapshot()):
            with self.assertRaises(SystemExit):runner.main()
        self.assertEqual([call.kwargs for call in client.request.call_args_list],[{'action':'start'},{'action':'stop'}])


if __name__=='__main__':unittest.main()
