"""Exercise real tracking guards with simulation I/O; hardware stays baseline."""
import threading
import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import numpy as np
from control_settle import wait_for_stable_control
from physical_core import PhysicalPlant
from simulation_tracking_limits import FIELDS, ENV_FIELDS, GLOBAL_ENV, environment_file, plant_class, profile
from sim_runtime import SimulationIO
from g1_can_handoff.sim import scene


class SimulationTrackingLimitsTests(unittest.TestCase):
    def plant(self, name):
        world = scene()
        io = SimulationIO(world, tracking_profile=name)
        config = {'arm_limits_rad': world.model.jnt_range[io.body_ids[15:]].tolist()}
        plant = plant_class(name)(scene(), io, io, io.read, config, Mock(), clock=io)
        return plant, io

    def test_double_profile_and_physical_defaults(self):
        old, new = profile('baseline'), profile('double')
        for field in (*FIELDS, 'motor_reference_leash_rad'):
            self.assertEqual(new[field], 2*old[field])
        self.assertEqual(PhysicalPlant.arm_tracking_error_rad, .15)
        self.assertEqual(PhysicalPlant.arm_settle_error_rad, .04)
        with self.assertRaises(ValueError):profile('triple')

    def test_environment_defaults_to_one_multiplier_for_every_joint(self):
        limits = profile('env', {})
        self.assertEqual(limits['global_multiplier'], 1.)
        for key, (field, count) in ENV_FIELDS.items():
            self.assertEqual(limits['multipliers'][key], [1.]*count)
            np.testing.assert_allclose(limits[field], profile('baseline')[field])

    def test_global_two_reaches_doubled_guards_for_every_field(self):
        limits = profile('env', {GLOBAL_ENV:'2'})
        for field in (*FIELDS, 'motor_reference_leash_rad'):
            np.testing.assert_allclose(limits[field], profile('double')[field])
        cls = plant_class('env', {GLOBAL_ENV:'2'})
        plant = cls.__new__(cls)
        plant._enforce_arm_tracking([.2]*7)
        with self.assertRaisesRegex(RuntimeError, '0.3 rad'):
            plant._enforce_arm_tracking([.31]*7)
        self.assertEqual(PhysicalPlant.arm_tracking_error_rad, .15)

    def test_global_composes_with_individual_joint_multipliers(self):
        limits = profile('env', {GLOBAL_ENV:'1.5', 'SIM_TRACKING_ARM_ERROR_MULTIPLIERS':'.5,1,1,1,1,1,1'})
        np.testing.assert_allclose(limits['arm_tracking_error_rad'], [.1125]+[.225]*6)
        self.assertEqual(limits['combined_multipliers']['SIM_TRACKING_ARM_ERROR_MULTIPLIERS'], [.75]+[1.5]*6)

    def test_global_and_combined_limits_fail_closed(self):
        for value in ('0','-1','nan','inf','2.1','1,1','garbage'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                profile('env', {GLOBAL_ENV:value})
        with self.assertRaisesRegex(ValueError, 'must not exceed 2'):
            profile('env', {GLOBAL_ENV:'2', 'SIM_TRACKING_ARM_ERROR_MULTIPLIERS':'1.5'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tracking.env'
            path.write_text(GLOBAL_ENV+'=1.5\n')
            self.assertEqual(profile('env', environment_file(path))['global_multiplier'], 1.5)

    def test_environment_enforces_each_joint_independently(self):
        env = {'SIM_TRACKING_ARM_ERROR_MULTIPLIERS': '.5,2,1,1,1,1,1'}
        plant = plant_class('env', env).__new__(plant_class('env', env))
        np.testing.assert_allclose(plant.arm_tracking_error_rad, [.075,.3,.15,.15,.15,.15,.15])
        self.assertAlmostEqual(plant._enforce_arm_tracking([0,.2,0,0,0,0,0]), .2)
        with self.assertRaisesRegex(RuntimeError, '0.075 rad at left_shoulder_pitch_joint'):
            plant._enforce_arm_tracking([.1,.2,0,0,0,0,0])
        # The baseline/physical class never reads these simulation variables.
        self.assertEqual(PhysicalPlant.arm_tracking_error_rad, .15)
        self.assertEqual(profile('baseline', env)['arm_tracking_error_rad'], .15)

    def test_multipliers_reject_invalid_values_and_wrong_joint_counts(self):
        for key in ENV_FIELDS:
            for value in ('0','-1','nan','inf','2.1','garbage','1,1'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    profile('env', {key:value})

    def test_explicit_environment_file_has_precedence_and_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tracking.env'
            path.write_text('# simulation only\nSIM_TRACKING_ARM_ERROR_MULTIPLIERS=1.5\n')
            limits = profile('env', environment_file(path))
            np.testing.assert_allclose(limits['arm_tracking_error_rad'], [.225]*7)
            path.write_text('HANDOFF_TRACKING_BYPASS=1\n')
            with self.assertRaisesRegex(ValueError, 'Invalid simulation tracking environment'):
                environment_file(path)

    def test_environment_profile_refuses_hardware_with_default_multipliers(self):
        writer = Mock()
        with self.assertRaisesRegex(ValueError, 'require SimulationIO'):
            plant_class('env', {})(None, writer, writer, Mock(), {}, Mock(), clock=writer)
        self.assertEqual(writer.mock_calls, [])

    def test_expanded_plant_refuses_hardware_before_construction(self):
        writer = Mock()
        with self.assertRaisesRegex(ValueError, 'require SimulationIO'):
            plant_class('double')(None, writer, writer, Mock(), {}, Mock(), clock=writer)
        writer.assert_not_called()
        self.assertEqual(writer.mock_calls, [])

    def test_hold_guard_in_all_phases_and_beyond_doubled_boundary(self):
        for phase in ('staging', 'approach', 'align', 'reach', 'close', 'lift', 'hold'):
            for name, error, accepted in (('baseline', .2, False), ('double', .2, True), ('double', .31, False)):
                with self.subTest(phase=phase, name=name, error=error):
                    plant, io = self.plant(name)
                    plant.set_contact_phase(phase)
                    snapshot = io.read()
                    snapshot['body']['q'][15] = float(plant.left_arm[0]+error)
                    plant.command_arm = Mock()
                    if accepted:
                        plant.hold_arm(snapshot)
                        plant.command_arm.assert_called_once()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'Measured arm tracking error'):
                            plant.hold_arm(snapshot)
                        plant.command_arm.assert_not_called()

    def test_execution_checks_requested_trajectory_and_settle_threshold(self):
        for name, accepted in (('baseline', False), ('double', True)):
            with self.subTest(name=name):
                plant, io = self.plant(name)
                goal = np.asarray(io.read()['body']['q'][15:22])
                reads = 0
                def read():
                    nonlocal reads
                    snapshot = io.read()
                    snapshot['body']['q'][15] = float(goal[0]+(.2 if reads == 0 else .06))
                    reads += 1
                    return snapshot
                plant.read = read
                plant.command_arm = Mock()
                if accepted:
                    plant._execute_poses(goal, [goal])
                    self.assertEqual(plant.command_arm.call_count, 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, '0.15 rad'):
                        plant._execute_poses(goal, [goal])
                    plant.command_arm.assert_not_called()

    def test_start_posture_guard_is_doubled_in_simulation_only(self):
        for name, accepted in (('baseline', False), ('double', True)):
            plant, io = self.plant(name)
            plant.path_preflight_ok = True
            plant.staging_start = np.asarray(io.read()['body']['q'][15:22])+.06
            if accepted:plant.validate_start()
            else:
                with self.assertRaisesRegex(RuntimeError, 'starting arm posture changed'):
                    plant.validate_start()
            self.assertEqual(io.commands, 0)

    def test_fk_guard_is_doubled_without_skipping_move(self):
        for name, accepted in (('baseline', False), ('double', True)):
            plant, io = self.plant(name)
            plant._move_poses = Mock(return_value=(np.zeros(7), []))
            plant._execute_poses = Mock()
            target = plant.ik.data.site_xpos[plant.ik.site].copy()+[.04, 0, 0]
            if accepted:plant.move(target)
            else:
                with self.assertRaisesRegex(RuntimeError, '25 mm'):
                    plant.move(target)
            plant._execute_poses.assert_called_once()

    def test_motor_reference_leash_is_doubled_but_speed_is_not(self):
        world = scene()
        io = SimulationIO(world, tracking_profile='double')
        io.ctrl_dual_arm(np.asarray(io.read()['body']['q'][15:29])+.4, np.zeros(14))
        io.sleep(.01)
        self.assertEqual(io.arm_reference.max_velocity, .25)
        self.assertEqual(io.arm_reference.max_acceleration, 1.)
        self.assertAlmostEqual(io.tracking_limits['motor_reference_leash_rad'], np.deg2rad(20))

    def test_doubled_profile_still_stops_stale_feedback_before_commands(self):
        plant, io = self.plant('double')
        def stale():
            snapshot = io.read()
            snapshot['body']['time'] -= 1.
            return snapshot
        plant.read = stale
        with self.assertRaisesRegex(ValueError, 'Stale body'):
            plant.hand(np.zeros(7))
        self.assertEqual(io.commands, 0)

    def test_doubled_profile_does_not_authorize_unplanned_staging(self):
        plant, io = self.plant('double')
        with self.assertRaisesRegex(RuntimeError, 'not been preflighted'):
            plant.stage()
        self.assertEqual(io.commands, 0)

    def test_arm_settling_span_scales_while_speed_guard_remains(self):
        class Clock:
            now = 100.
            def monotonic(self):return self.now
            def sleep(self, seconds):self.now += seconds
        for limit, accepted in ((.015, False), (.03, True)):
            clock = Clock()
            def read():
                s = {side:dict(q=[0.]*count, dq=[0.]*count, tau_est=[0.]*count,
                              time=clock.now, temperature_max=25., faults=[0]*count, imu_rpy=[0.]*3)
                     for side, count in (('body', 29), ('left', 7), ('right', 7))}
                s['body']['q'][15] = .011*np.sin((clock.now-100)*20)
                return s
            if accepted:
                wait_for_stable_control(read, threading.Event(), Mock(), Mock(), clock, arm_span_limit_rad=limit)
            else:
                with self.assertRaisesRegex(RuntimeError, 'within 5 seconds'):
                    wait_for_stable_control(read, threading.Event(), Mock(), Mock(), clock, arm_span_limit_rad=limit)


if __name__ == '__main__':unittest.main()
