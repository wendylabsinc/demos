import unittest
from operator_state import describe_state


class OperatorStateTests(unittest.TestCase):
    def test_hold_is_visibly_distinct_from_released_even_with_same_error(self):
        state=dict(phase='FROZEN',adapter_initialized=True,demo_running=False,error='Collision')
        held=describe_state(state)
        self.assertEqual(held['tone'],'hold');self.assertIn('arms held',held['title'])
        self.assertIn('Support',held['next_action'])
        state.update(phase='PASSIVE',adapter_initialized=False)
        released=describe_state(state)
        self.assertEqual(released['tone'],'ready');self.assertEqual(released['failure_label'],'Last attempt')
        self.assertIn('Unitree',released['ownership'])

    def test_previous_verified_lift_does_not_claim_demo_holds_after_release(self):
        state=dict(phase='PASSIVE',adapter_initialized=False,demo_running=False,
                   demo_result=dict(physical_lift_verified=True))
        self.assertEqual(describe_state(state)['tone'],'ready')

    def test_planning_motion_and_finger_closing_have_distinct_labels(self):
        for phase,title in [('grasp_candidate_rejected','Planning pickup'),
                            ('CONTROL_SETTLING','Taking arm control'),
                            ('staging_segment','Moving to pickup'),('close','Closing fingers')]:
            state=dict(phase=phase,adapter_initialized=True,demo_running=True)
            self.assertEqual(describe_state(state)['title'],title)

    def test_unknown_ownership_and_unhealthy_feedback_are_not_ready(self):
        self.assertEqual(describe_state({})['tone'],'unknown')
        self.assertEqual(describe_state(dict(phase='PASSIVE',adapter_initialized=False,
            telemetry_healthy=False))['tone'],'blocked')

    def test_previous_error_is_not_a_current_problem_during_another_attempt(self):
        state=describe_state(dict(phase='PLANNING',adapter_initialized=True,demo_running=True,
                                 demo_error='Previous attempt failed'))
        self.assertIsNone(state['failure'])
        self.assertEqual(state['activity'],'Attempt running')

    def test_stop_phase_wins_over_worker_running_flag_during_cleanup(self):
        state=describe_state(dict(phase='FROZEN',adapter_initialized=True,demo_running=True,
                                 demo_error='Color/depth can target absent/ambiguous'))
        self.assertEqual(state['title'],'Stopped — arms held')
        self.assertEqual(state['activity'],'Attempt stopped')
        self.assertNotIn('running',state['next_action'])

    def test_alignment_and_reach_are_not_shown_as_planning(self):
        for phase,title in [('approach','Approaching the can'),('align','Aligning the hand'),
                            ('reach','Reaching the can')]:
            with self.subTest(phase=phase):
                self.assertEqual(describe_state(dict(phase=phase,adapter_initialized=True,
                    demo_running=True))['title'],title)


if __name__=='__main__':unittest.main()
