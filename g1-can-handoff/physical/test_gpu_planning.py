"""FK parity, unsafe intermediate poses, fail-closed CUDA and path reuse."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import mujoco
from test_arm_gravity import ArmGravityTests

try:
    import warp as wp
    from gpu_kinematics import ArmKinematics, Vec7
    from gpu_planner import (
        CudaPlanner,
        PHASES,
        contact_thresholds,
        contacts_kernel,
        joints_kernel,
    )
except ImportError:
    wp = None


@unittest.skipIf(
    wp is None, "Install requirements-gpu.txt for kernel development tests"
)
class GpuKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plant, _, _, _ = ArmGravityTests().plant()
        cls.arm = ArmKinematics(cls.plant.ik, "cpu")

    def test_fk_matches_commissioned_model_over_joint_range(self):
        ik = self.plant.ik
        limits = np.asarray(self.plant.config["arm_limits_rad"])[:7]
        qs = np.random.default_rng(8).uniform(limits[:, 0], limits[:, 1], (100, 7))
        positions, orientations = self.arm.forward(qs)
        data = mujoco.MjData(ik.model)
        data.qpos[:] = ik.data.qpos
        for index, q in enumerate(qs):
            data.qpos[ik.qadr[ik.arm]] = q
            mujoco.mj_kinematics(ik.model, data)
            np.testing.assert_allclose(
                positions[index], data.site_xpos[ik.site], rtol=0, atol=5e-7
            )
            np.testing.assert_allclose(
                orientations[index],
                data.site_xmat[ik.site].reshape(3, 3),
                rtol=0,
                atol=7e-7,
            )

    def test_ik_continues_same_branch_and_rejects_unreachable_target(self):
        ik = self.plant.ik
        start = ik.data.qpos[ik.qadr[ik.arm]].copy()
        p, r = self.arm.forward(start)
        limits = np.asarray(self.plant.config["arm_limits_rad"])[:7]
        goal = self.arm.solve(p[0] + [0, 0, 0.005], r[0], start, limits, local=True)
        self.assertLessEqual(np.max(np.abs(goal - start)), 0.200001)
        np.testing.assert_allclose(
            self.arm.forward(goal)[0][0], p[0] + [0, 0, 0.005], atol=1e-4
        )
        with self.assertRaisesRegex(RuntimeError, "cannot preserve"):
            self.arm.solve([10, 10, 10], r[0], start, limits, local=True)

    def test_gpu_joint_checker_rejects_raw_angle_flip_between_safe_endpoints(self):
        q = np.zeros((3, 7), np.float32)
        q[1, 0] = 2 * np.pi
        invalid = wp.zeros(3, dtype=int, device="cpu")
        wp.launch(
            joints_kernel,
            3,
            [
                wp.array(q, dtype=Vec7, device="cpu"),
                3,
                Vec7(*([-7.0] * 7)),
                Vec7(*([7.0] * 7)),
                invalid,
            ],
            device="cpu",
        )
        self.assertEqual(invalid.numpy().tolist(), [0, 2, 2])

    def test_gpu_contact_checker_catches_middle_pose_table_collision(self):
        model = self.plant.ik.model
        table = model.geom("table").id
        wrist = next(
            i
            for i in range(model.ngeom)
            if model.geom_contype[i]
            and model.body(model.geom_bodyid[i]).name == "left_wrist_yaw_link"
        )
        self.plant.ik.table_clearance_m = 0.01
        invalid = wp.zeros(3, dtype=int, device="cpu")
        wp.launch(
            contacts_kernel,
            3,
            [
                wp.array([3], dtype=int, device="cpu"),
                wp.array([0, 1, 2], dtype=int, device="cpu"),
                wp.array([[table, wrist]] * 3, dtype=wp.vec2i, device="cpu"),
                wp.array([0.02, -0.005, 0.02], dtype=float, device="cpu"),
                wp.array([PHASES.index("approach")] * 3, dtype=int, device="cpu"),
                wp.array(contact_thresholds(self.plant.ik), dtype=float, device="cpu"),
                3,
                invalid,
            ],
            device="cpu",
        )
        self.assertEqual(invalid.numpy().tolist(), [0, 4, 0])

    def test_gpu_policy_matches_cpu_for_every_geom_pair_and_phase(self):
        ik = self.plant.ik
        thresholds = contact_thresholds(ik)
        old = ik.contact_phase
        try:
            for phase, name in enumerate(PHASES):
                ik.set_contact_phase(name)
                for a in range(ik.model.ngeom):
                    for b in range(ik.model.ngeom):
                        for distance in (-0.02, -0.0005, 0.005, 0.015):
                            contact = SimpleNamespace(geom1=a, geom2=b, dist=distance)
                            expected = (
                                ik.collision_details(SimpleNamespace(contact=[contact]))
                                is not None
                            )
                            self.assertEqual(
                                distance < thresholds[phase, a, b],
                                expected,
                                (name, a, b, distance),
                            )
        finally:
            ik.set_contact_phase(old)

    def test_missing_cuda_cannot_fall_back_to_cpu(self):
        with patch("gpu_planner.wp.is_cuda_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                CudaPlanner(self.plant.ik)


class PreparedPathTests(unittest.TestCase):
    def setUp(self):
        self.plant, self.adapter, _, _ = ArmGravityTests().plant()
        p = self.plant
        p.planning_backend = "cuda"
        p.path_preflight_ok = True
        p.set_contact_phase("approach")
        start = p.ik.data.qpos[p.ik.qadr[p.ik.arm]].copy()
        self.target = p.ik.data.site_xpos[p.ik.site].copy()
        self.rotation = p.ik.data.site_xmat[p.ik.site].reshape(3, 3).copy()
        p.prepared_steps = dict(
            approach=dict(
                start=start,
                goal=start.copy(),
                poses=[start.copy()],
                position=self.target,
                rotation=self.rotation,
            )
        )
        p.prepared_scene = p._scene_stamp()
        p._execute_poses = Mock()
        p._move_poses = Mock(
            side_effect=AssertionError("Must not re-solve the prepared path")
        )

    def test_execution_reuses_validated_samples_without_ik_or_path_check(self):
        self.plant.move(self.target, rotation=self.rotation)
        self.plant._execute_poses.assert_called_once()
        self.plant._move_poses.assert_not_called()

    def test_target_or_scene_change_aborts_before_any_command(self):
        with self.assertRaisesRegex(RuntimeError, "target or starting"):
            self.plant.move(self.target + [0, 0, 0.01], rotation=self.rotation)
        self.plant.ik.model.geom("table").pos[2] += 0.01
        with self.assertRaisesRegex(RuntimeError, "Fixed scene geometry"):
            self.plant.move(self.target, rotation=self.rotation)
        self.plant._execute_poses.assert_not_called()
        self.adapter.ctrl_dual_arm.assert_not_called()

    def test_cuda_rejects_unprepared_movement_instead_of_running_cpu_ik(self):
        self.plant.planning_backend = "cuda"
        self.plant.path_preflight_ok = False
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            self.plant.move(self.target, rotation=self.rotation)
        self.plant._move_poses.assert_not_called()
        self.plant._execute_poses.assert_not_called()


class SingleCudaPlanTests(unittest.TestCase):
    def test_one_accepted_path_is_not_ranked_or_revalidated(self):
        from g1_can_handoff.task import plan

        observation = SimpleNamespace(
            center=np.array([0.31, 0.15, 0.823]), table_height=0.762, timestamp=100.0
        )
        plant = SimpleNamespace(
            planning_backend="cuda",
            grasp_candidate_budget=3,
            set_contact_phase=Mock(),
            grasp_noise_score=Mock(side_effect=AssertionError("No ranking")),
        )
        validate = Mock(return_value=True)
        with patch("g1_can_handoff.task.time.monotonic", return_value=100.0):
            plan(plant, observation, validate=validate, include_offer=False)
        validate.assert_called_once()
        plant.grasp_noise_score.assert_not_called()

    def test_unsafe_cuda_path_returns_without_trying_more_complete_paths(self):
        from g1_can_handoff.task import plan

        observation = SimpleNamespace(
            center=np.array([0.31, 0.15, 0.823]), table_height=0.762, timestamp=100.0
        )
        plant = SimpleNamespace(planning_backend="cuda", set_contact_phase=Mock())
        validate = Mock(return_value=False)
        with patch("g1_can_handoff.task.time.monotonic", return_value=100.0):
            with self.assertRaisesRegex(ValueError, "path rejected"):
                plan(plant, observation, validate=validate, include_offer=False)
        validate.assert_called_once()

    def test_cuda_deadline_invalidates_a_late_plan_before_any_motion(self):
        from g1_can_handoff.task import plan

        clock = {"time": 100.0}
        observation = SimpleNamespace(
            center=np.array([0.31, 0.15, 0.823]), table_height=0.762, timestamp=100.0
        )
        plant = SimpleNamespace(
            planning_backend="cuda", set_contact_phase=Mock(), path_preflight_ok=True
        )

        def slow(*args):
            clock["time"] += 1.001
            return True

        with patch(
            "g1_can_handoff.task.time.monotonic", side_effect=lambda: clock["time"]
        ):
            with self.assertRaisesRegex(ValueError, "search time limit"):
                plan(plant, observation, validate=slow, include_offer=False)
        self.assertFalse(plant.path_preflight_ok)


if __name__ == "__main__":
    unittest.main()
