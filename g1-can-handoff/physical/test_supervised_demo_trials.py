import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supervised_demo_trials import preflight, reserve, run


class SupervisedTrialTests(unittest.TestCase):
    def test_no_automatic_second_start_without_operator_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'trial-01.json').write_text(json.dumps(dict(demo_running_after=False)))
            with self.assertRaisesRegex(RuntimeError,'reset confirmation'):
                reserve(root,2,False)
            self.assertFalse((root/'trial-02.json').exists())
            self.assertTrue(reserve(root,2,True).exists())

    def test_reset_acknowledgement_cannot_skip_a_running_previous_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'trial-01.json').write_text(json.dumps(dict(demo_running_after=True)))
            with self.assertRaisesRegex(RuntimeError,'terminal trial'):
                reserve(root,2,True)

    def test_duplicate_run_never_constructs_transport_or_reissues_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reserve(root,1,False)
            with patch('supervised_demo_trials.DEPLOYING') as marker, patch('supervised_demo_trials.Wendy') as client:
                marker.is_file.return_value=False
                with self.assertRaisesRegex(RuntimeError,'already reserved'):
                    run(root,1,False)
                client.assert_not_called()

    def test_retained_ownership_requires_release_even_if_reset_was_acknowledged(self):
        for phase,owned,running in [('FROZEN',True,False),('PASSIVE',True,False),('PASSIVE',False,True)]:
            with self.subTest(phase=phase,owned=owned,running=running):
                before=dict(status=dict(phase=phase,adapter_initialized=owned,demo_running=running),recorder={})
                with self.assertRaisesRegex(RuntimeError,'Reset/release required'):
                    preflight(before)


if __name__=='__main__':unittest.main()
