"""Exact compiled-model replay, failures, provenance and measured velocity seeds."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from g1_can_handoff.sim import scene,FINGERS
from sim_runtime import SimulationIO,apply_measured_state
from physical_core import PhysicalPlant,BODY_NAMES,SIM_TO_BRIDGE
from planning_snapshot import PlanningRecorder,replay


class PlanningSnapshotTests(unittest.TestCase):
    def test_full_model_and_mutable_state_replay_both_ik_and_failed_path(self):
        world=scene();io=SimulationIO(world)
        config=dict(arm_limits_rad=world.model.jnt_range[io.body_ids[15:]].tolist())
        plant=PhysicalPlant(scene(),None,None,io.read,config,lambda *a,**k:None,clock=io)
        with tempfile.TemporaryDirectory() as directory:
            plant.planning_recorder=PlanningRecorder(directory,lambda *a,**k:None)
            # This is an actual FK pose; no recorded IK goal is used as input.
            position=plant.ik.data.site_xpos[plant.ik.site].copy()
            rotation=plant.ik.data.site_xmat[plant.ik.site].reshape(3,3).copy()
            plant.solve(position,rotation)
            path=next(Path(directory).glob('*-ik.json'))
            captured=json.loads(path.read_text())
            self.assertEqual(captured['measured'],plant.last_synced_snapshot)
            self.assertEqual(len(captured['qpos']),plant.ik.model.nq)
            self.assertEqual(captured['solver_settings']['ik']['max_nfev'],250)
            self.assertTrue(replay(path)['identical'])
            self.assertEqual(replay(path)['max_abs_error_rad'],0.)
            # Mutate a model pose after the cached MJB was saved.
            plant.ik.model.body_pos[plant.ik.model.body('pelvis').id,0]+=.001
            import mujoco
            mujoco.mj_forward(plant.ik.model,plant.ik.data)
            plant.ik.solve(plant.ik.data.site_xpos[plant.ik.site].copy(),
                           plant.ik.data.site_xmat[plant.ik.site].reshape(3,3).copy())
            self.assertTrue(all(replay(p)['identical'] for p in Path(directory).glob('*-ik.json')))
            with self.assertRaisesRegex(RuntimeError,'exceeds commissioned'):
                plant._joint_poses(np.full(7,99.))
            failed=next(Path(directory).glob('*-joint_poses.json'))
            self.assertTrue(replay(failed)['identical'])
            self.assertEqual(len(list(Path(directory).glob('*.mjb'))),1)
            model_path=next(Path(directory).glob('*.mjb'))
            model_path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):replay(path)

    def test_seed_preserves_every_body_and_asymmetric_hand_velocity(self):
        world=scene();io=SimulationIO(world);measured=io.read()
        for side,count in [('body',29),('left',7),('right',7)]:
            measured[side]['dq']=np.linspace(-.03,.04,count).tolist()
        apply_measured_state(world,io,measured)
        np.testing.assert_array_equal(world.data.qvel[world.model.jnt_dofadr[io.body_ids]],measured['body']['dq'])
        for side in ('left','right'):
            expected=np.asarray(measured[side]['dq'])
            if side=='left':expected=expected[SIM_TO_BRIDGE]
            np.testing.assert_array_equal(world.data.qvel[world.model.jnt_dofadr[io.hand_ids[side]]],expected)


if __name__=='__main__':unittest.main()
