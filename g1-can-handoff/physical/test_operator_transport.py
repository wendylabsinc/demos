"""Failed Wendy connection must not leak its local MCP subprocess."""

import unittest
import contextlib
import io
import json
import threading
from unittest.mock import Mock, patch

from operator_panel import Wendy


class OperatorTransportTests(unittest.TestCase):
    def test_status_keeps_reported_hand_faults_and_sensor_pairs(self):
        """Exercise the actual remote projection that previously dropped feedback."""
        sample = {'phase':'PASSIVE','adapter_initialized':False,'telemetry_healthy':False,
                  'measured':{'left':{'faults':[0,512,512,0,0,0,0,0,16],
                                      'temperatures':[[38,37]]*7,'time':10.0}}}
        client = Wendy.__new__(Wendy)
        client.lock = threading.Lock()
        def remote_tool(name, args):
            self.assertEqual(name, 'container_exec')
            captured = io.StringIO()
            with (patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(sample).encode())),
                  patch('time.monotonic', return_value=10.1), contextlib.redirect_stdout(captured)):
                exec(args['command'][2], {})
            return {'structuredContent':{'exit_code':0,'stdout':captured.getvalue(),'truncated':False}}
        client.tool = remote_tool
        result = client.request()
        self.assertEqual(result['hand_feedback']['left']['faults'], [0,512,512,0,0,0,0,0,16])
        self.assertEqual(result['hand_feedback']['left']['temperatures'], [[38,37]]*7)
        self.assertAlmostEqual(result['hand_feedback']['left']['age_ms'], 100)
        self.assertIsNone(result['hand_feedback']['right']['faults'])
        self.assertIsNone(result['hand_feedback']['right']['age_ms'])

    def test_failed_device_connection_terminates_created_mcp_process(self):
        process = Mock()
        process.stdout = iter(())
        with (
            patch("operator_panel.subprocess.Popen", return_value=process),
            patch.object(Wendy, "rpc", return_value={}),
            patch.object(
                Wendy, "tool", side_effect=[{}, RuntimeError("DEVICE_UNREACHABLE")]
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "DEVICE_UNREACHABLE"):
                Wendy()
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
