"""A recorded planner stop is reproduced with a controlled clock, no 30s sleep."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from g1_can_handoff.task import plan
from g1_can_handoff.errors import IKError

class PlanningDeadlineTests(unittest.TestCase):
    def test_recorded_orientation_timeout_is_bounded_before_any_motion(self):
        trace=json.loads(gzip.decompress((Path(__file__).parent/'offline/evidence/latest-planner-stop.json.gz').read_bytes()))
        failure=next(e for e in trace['events'] if e['phase']=='primary_failure')
        self.assertEqual(failure['error'],'ValueError: Grasp orientation search time limit reached')
        self.assertFalse(trace['status']['adapter_initialized'])
        self.assertFalse(any(e['phase']=='target' for e in trace['events']))
        state={'time':100.}
        plant=Mock()
        def blocked_candidate(*args):
            # Reproduce expensive failed IK/preflight, advancing virtual time.
            state['time']+=30.001
            raise IKError('No candidate IK solution')
        plant.solve.side_effect=blocked_candidate
        observation=SimpleNamespace(center=np.array([.35,.18,.823]),table_height=.762,timestamp=100.)
        with patch('g1_can_handoff.task.time.monotonic',side_effect=lambda:state['time']):
            with self.assertRaisesRegex(ValueError,'Grasp orientation search time limit reached'):
                plan(plant,observation,include_offer=False)
        self.assertEqual(plant.solve.call_count,1)
        plant.move.assert_not_called();plant.hand.assert_not_called()

if __name__=='__main__':unittest.main()
