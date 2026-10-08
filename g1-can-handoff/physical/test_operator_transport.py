"""Failed Wendy connection must not leak its local MCP subprocess."""
import unittest
from unittest.mock import Mock,patch

from operator_panel import Wendy


class OperatorTransportTests(unittest.TestCase):
    def test_failed_device_connection_terminates_created_mcp_process(self):
        process=Mock();process.stdout=iter(())
        with patch('operator_panel.subprocess.Popen',return_value=process),\
             patch.object(Wendy,'rpc',return_value={}),\
             patch.object(Wendy,'tool',side_effect=[{},RuntimeError('DEVICE_UNREACHABLE')]):
            with self.assertRaisesRegex(RuntimeError,'DEVICE_UNREACHABLE'):Wendy()
        process.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
