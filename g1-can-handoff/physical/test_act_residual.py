"""Numerical ACT parity and robot-adapter regressions; motor IO is MuJoCo only."""
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import numpy as np
from act_numpy import NumpyACT, verify_parity
from act_residual import ACTResidual, actor_observation
from physical_core import PhysicalPlant
from g1_can_handoff.sim import scene
from sim_runtime import SimulationIO

DIRECTORY = Path(__file__).parent / 'act_policy'


class ACTTests(unittest.TestCase):
    def plant(self):
        world = scene()
        io = SimulationIO(world)
        io.sleep(.02)  # Populate the simulation's motor-clock reference without publishing.
        config = dict(arm_limits_rad=world.model.jnt_range[io.body_ids[15:]].tolist())
        events = []
        plant = PhysicalPlant(scene(), io, io, io.read, config,
                              lambda name, **fields: events.append((name, fields)), clock=io,
                              cancel=threading.Event())
        model = NumpyACT(DIRECTORY)
        policy = ACTResidual(model, plant, lambda: (model.metadata['can_initial'], io.monotonic()))
        plant.act_residual = policy
        plant.set_contact_phase('approach')
        return plant, io, policy, events

    def test_selected_checkpoint_matches_1000_pytorch_observations(self):
        self.assertTrue(verify_parity(DIRECTORY)['passed'])

    def test_temporal_ensemble_matches_original_age_weighting(self):
        model = NumpyACT(DIRECTORY)
        fixture = np.load(DIRECTORY / 'parity-fixtures.npz')
        history = []
        for obs, chunk in zip(fixture['observations'][:40], fixture['chunks'][:40]):
            history.append(chunk)
            history = history[-model.chunk_size:]
            weights = np.exp(-.05 * np.arange(len(history), dtype=np.float32))
            expected = (np.stack([c[age] for age, c in enumerate(reversed(history))]) * weights[:, None]).sum(0) / weights.sum()
            np.testing.assert_allclose(model.predict(obs), expected, atol=2e-5, rtol=2e-5)

    def test_observation_layout_is_identical_to_simulator_contract(self):
        q = np.arange(28, dtype=float)
        obs = actor_observation(q, q + 100, q[:7] + 200,
             [1, 2, 3], [4, 5, 6], [7, 8, 9], 2, 4, np.arange(5), True, q[14:21] + .5)
        np.testing.assert_array_equal(obs[:28], q)
        np.testing.assert_array_equal(obs[28:56], q + 100)
        np.testing.assert_array_equal(obs[56:63], q[:7] + 200)
        np.testing.assert_array_equal(obs[63:72], np.arange(1, 10))
        self.assertEqual(obs[72], 2)
        np.testing.assert_array_equal(obs[73:81], np.eye(8)[4])
        np.testing.assert_array_equal(obs[81:86], np.arange(5))
        self.assertEqual(obs[86], 1)
        np.testing.assert_array_equal(obs[87:], [.5] * 7)
        with self.assertRaises(ValueError):
            actor_observation(q, q, q[:7], [1, 2, np.nan], [4, 5, 6], [7, 8, 9], 0, 0, np.zeros(5), False, q[14:21])

    def test_hold_does_not_ratchet_residual_into_the_nominal_target(self):
        plant, io, policy, events = self.plant()
        nominal = plant.left_arm.copy()
        for _ in range(50):
            plant.maintain_hand()
            io.sleep(.02)
            np.testing.assert_array_equal(policy.nominal_target, nominal)
            self.assertLessEqual(np.max(np.abs(plant.left_arm - nominal)), .1 + 1e-9)
            self.assertLessEqual(np.max(np.abs(io.arm_reference.velocity)), .25 + 1e-9)
        self.assertGreater(policy.observation_count, 5)
        self.assertGreater(io.commands, 0)

    def test_stale_policy_pose_fails_before_any_simulated_motor_command(self):
        plant, io, policy, events = self.plant()
        policy.pose_supplier = lambda: (policy.model.metadata['can_initial'], io.monotonic() - 1)
        with self.assertRaisesRegex(RuntimeError, 'exposure stale'):
            plant.maintain_hand()
        self.assertEqual(io.commands, 0)

    def test_inference_between_20_and_40_ms_is_admitted_without_motor_io(self):
        plant, io, policy, events = self.plant()
        now = io.monotonic()
        snapshot = io.read()
        policy.pose_supplier = lambda: (policy.model.metadata['can_initial'], now)
        with patch.object(io, 'monotonic', side_effect=[now, now, now + .030]):
            target = policy.update(plant.left_arm, snapshot)
        self.assertTrue(np.isfinite(target).all())
        self.assertEqual(policy.observation_count, 1)
        self.assertEqual(io.commands, 0)

    def test_inference_over_40_ms_fails_before_any_motor_io(self):
        plant, io, policy, events = self.plant()
        now = io.monotonic()
        snapshot = io.read()
        policy.pose_supplier = lambda: (policy.model.metadata['can_initial'], now)
        with patch.object(io, 'monotonic', side_effect=[now, now, now + .041]):
            with self.assertRaisesRegex(RuntimeError, '40 ms control budget'):
                policy.update(plant.left_arm, snapshot)
        self.assertEqual(policy.observation_count, 0)
        self.assertEqual(io.commands, 0)

    def test_invalid_inference_budget_is_rejected_before_adapter_activation(self):
        for value in ('0', '-1', 'nan', 'inf', '101', 'not-a-number'):
            with self.subTest(value=value), patch.dict('os.environ', {'HANDOFF_ACT_INFERENCE_BUDGET_MS': value}):
                with self.assertRaisesRegex(ValueError, 'inference budget'):
                    self.plant()

    def test_invalid_velocity_fails_before_any_simulated_motor_command(self):
        plant, io, policy, events = self.plant()
        read = plant.read
        def invalid():
            snapshot = read()
            snapshot['left']['dq'][0] = np.nan
            return snapshot
        plant.read = invalid
        with self.assertRaises(ValueError):
            plant.maintain_hand()
        self.assertEqual(io.commands, 0)

    def test_grasp_latches_can_without_any_further_camera_access(self):
        plant, io, policy, events = self.plant()
        plant.hold_arm(io.read())
        retained = policy.last_can.copy()
        plant.grip = SimpleNamespace(grasped=True)
        policy.pose_supplier = lambda: (_ for _ in ()).throw(AssertionError('Postgrasp camera access'))
        plant.set_contact_phase('lift')
        io.sleep(.12)
        plant.hold_arm(io.read())
        io.sleep(.12)
        plant.hold_arm(io.read())
        np.testing.assert_array_equal(policy.last_can, retained)
        self.assertEqual(policy.last_observation[86], 1)
        self.assertGreater(policy.last_observation[72], 0)

    def test_freeze_disables_policy_and_blocks_further_commands(self):
        plant, io, policy, events = self.plant()
        plant.freeze()
        self.assertFalse(policy.active)
        count = io.commands
        with self.assertRaises(RuntimeError):
            plant.maintain_hand()
        self.assertEqual(io.commands, count)

    def test_domain_admission_rejects_unqualified_placement(self):
        plant, io, policy, events = self.plant()
        origin = np.asarray(policy.model.metadata['can_initial'])
        policy.validate_can(origin)
        for delta in ([.011, 0, 0], [0, .011, 0], [0, 0, .02]):
            with self.assertRaises(ValueError):
                policy.validate_can(origin + delta)


if __name__ == '__main__':
    unittest.main()
