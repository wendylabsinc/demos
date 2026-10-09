"""Integration coverage using real MuJoCo dynamics and physical control code."""
import threading
import unittest
import numpy as np
from physical_core import PhysicalPlant
from sim_runtime import SimulationIO, run
from g1_can_handoff.sim import scene


class PhysicalSimulationTests(unittest.TestCase):
    def plant(self):
        world=scene();io=SimulationIO(world);cancel=threading.Event()
        config={'arm_limits_rad':world.model.jnt_range[io.body_ids[15:]].tolist()}
        plant=PhysicalPlant(scene(),io,io,io.read,config,lambda *a,**kw:None,clock=io,cancel=cancel)
        return plant,io,cancel

    def test_arm_servo_uses_commanded_torque_without_hidden_gravity(self):
        world=scene();io=SimulationIO(world)
        q=np.asarray(io.read()['body']['q'][15:])
        io.ctrl_dual_arm(q,np.zeros(14));io.sleep(.002)
        np.testing.assert_allclose(world.data.ctrl[io.arm_ids],0,atol=1e-12)
        world=scene();io=SimulationIO(world)
        q=np.asarray(io.read()['body']['q'][15:])
        feedforward=np.linspace(-.3,.3,14)
        io.ctrl_dual_arm(q,feedforward);io.sleep(.002)
        np.testing.assert_allclose(world.data.ctrl[io.arm_ids],feedforward,atol=1e-12)
        np.testing.assert_array_equal(io.arm_kp,[150,150,150,150,40,40,40]*2)
        np.testing.assert_array_equal(io.arm_kd,[3,3,3,3,1.5,1.5,1.5]*2)

    def test_arm_servo_reuses_bounded_adapter_reference(self):
        world=scene();io=SimulationIO(world)
        q=np.asarray(io.read()['body']['q'][15:]);target=q+.1
        io.ctrl_dual_arm(target,np.zeros(14));io.sleep(.01)
        self.assertLess(np.max(np.abs(io.wire_arm_q-q)),.001)
        self.assertLessEqual(np.max(np.abs(io.arm_reference.velocity)),.25)

    def test_cancelled_hand_open_sends_no_commands(self):
        plant,io,cancel=self.plant();cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):plant.hand(np.zeros(7))
        self.assertEqual(io.commands,0)

    def test_stale_hand_feedback_sends_no_commands(self):
        plant,io,cancel=self.plant();read=plant.read
        def stale():
            snapshot=read();snapshot['left']['time']-=1
            return snapshot
        plant.read=stale
        with self.assertRaisesRegex(ValueError,'Stale left'):plant.hand(np.zeros(7))
        self.assertEqual(io.commands,0)

    def test_freeze_prevents_subsequent_watchdog_hand_command(self):
        plant,io,cancel=self.plant();plant.freeze();count=io.commands
        with self.assertRaisesRegex(RuntimeError,'cancelled'):plant.maintain_hand()
        self.assertEqual(io.commands,count)

    def test_freeze_serializes_with_inflight_command(self):
        plant,io,cancel=self.plant()
        entered=threading.Event();release=threading.Event();frozen=threading.Event()
        original=io.ctrl_dual_hand
        def blocked(*args):
            entered.set()
            if not release.wait(2):raise RuntimeError('Test command did not unblock')
            original(*args)
        io.ctrl_dual_hand=blocked
        command=threading.Thread(target=plant.maintain_hand)
        stopper=threading.Thread(target=lambda:(plant.freeze(),frozen.set()))
        command.start()
        self.assertTrue(entered.wait(2))
        stopper.start()
        try:self.assertFalse(frozen.wait(.02))
        finally:release.set();command.join(2);stopper.join(2)
        self.assertTrue(frozen.is_set())
        count=io.commands
        with self.assertRaisesRegex(RuntimeError,'cancelled'):plant.maintain_hand()
        self.assertEqual(io.commands,count)

    def test_staging_rejects_changed_measured_start_before_commands(self):
        plant,io,cancel=self.plant()
        plant.path_preflight_ok=True
        plant.staging_start=np.asarray(io.read()['body']['q'][15:22])+.05
        with self.assertRaisesRegex(RuntimeError,'starting arm posture changed'):plant.stage()
        self.assertEqual(io.commands,0)

    def test_staging_requires_completed_preflight(self):
        plant,io,cancel=self.plant()
        plant.staging_start=np.asarray(io.read()['body']['q'][15:22])
        with self.assertRaisesRegex(RuntimeError,'not been preflighted'):plant.stage()
        self.assertEqual(io.commands,0)

    def test_shared_physical_sequence_lifts_with_measured_table_height(self):
        result=run(position_noise='none',height=.762)
        self.assertTrue(result['lift_verified'],result.get('error'))
        self.assertGreater(result['visual_rise_m'],.11)
        self.assertEqual(result['hold_verified_s'],1.)
        self.assertFalse(result['physical_pickup_verified'])
        self.assertEqual(result['backend_assumptions']['hand_kp'],2.)
        self.assertGreater(result['command_count'],0)
        phases=[e['phase'] for e in result['events']]
        self.assertIn('contact_proxy',phases)
        self.assertLess(phases.index('control_settled'),phases.index('plan'))
        self.assertEqual(phases.count('plan'),1)
        self.assertEqual(phases[-1],'lift_verified')

    def test_measured_arms_down_start_rejects_interpolated_collision(self):
        initial=[.246,.265,-.092,.820,.0029,.00794,-.00446,
                 .265,-.286,.089,.798,-.00014,.0102,.00156]
        result=run(position_noise='none',height=.762,initial_arm_q14=initial,staging_enabled=False)
        self.assertFalse(result['lift_verified'])
        self.assertIn('trajectory intersects',result['error'])
        self.assertNotIn('target',result['event_counts'])
        # Taking ownership now issues a measured hold before planning; rejected
        # paths must still issue no pickup trajectory or finger-open commands.
        self.assertGreater(result['command_count'],0)
        self.assertNotIn('hand_target',result['event_counts'])

    def test_measured_arms_down_stages_then_lifts_and_holds(self):
        initial=[.246,.265,-.092,.820,.0029,.00794,-.00446,
                 .265,-.286,.089,.798,-.00014,.0102,.00156]
        result=run(position_noise='none',height=.762,initial_arm_q14=initial)
        self.assertTrue(result['lift_verified'],result.get('error'))
        self.assertEqual(result['hold_verified_s'],1.)
        planned=[e for e in result['events'] if e['phase']=='staging_planned'][-1]['joint_waypoints']
        executed=[e['goal_q'] for e in result['events'] if e['phase']=='staging_segment']
        self.assertGreater(len(executed),0)
        np.testing.assert_allclose(executed,planned)
        self.assertEqual(result['simulation_contact_audit']['pregrasp_contact_steps'],0)
        self.assertEqual(result['event_counts']['lift_observation'],5)
        self.assertLess(result['max_tracking_error_rad'],.15)

    def test_75cm_scene_tracks_finger_split_label_during_hold(self):
        result=run(position_noise='none',height=.75)
        self.assertTrue(result['lift_verified'],result.get('error'))
        self.assertEqual(result['event_counts']['lift_observation'],5)

    def test_head_camera_tracks_finger_split_label_during_hold(self):
        result=run(position_noise='none',camera='head')
        self.assertTrue(result['lift_verified'],result.get('error'))
        self.assertEqual(result['event_counts']['lift_observation'],5)
        self.assertFalse(result['physical_pickup_verified'])

if __name__=='__main__':unittest.main()
