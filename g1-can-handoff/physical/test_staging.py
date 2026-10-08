"""Joint planner bounds, cancellation and immutable measured model state."""
import unittest
import numpy as np
import mujoco
from g1_can_handoff.sim import scene
from staging import joint_path


class StagingPlannerTests(unittest.TestCase):
    def setUp(self):
        self.plant=scene(height=.762)
        self.limits=self.plant.model.jnt_range[self.plant.joints[self.plant.arm]].copy()
        self.start=self.plant.data.qpos[self.plant.qadr[self.plant.arm]].copy()

    def test_starting_model_is_preserved(self):
        before=self.plant.data.qpos.copy()
        path,meta=joint_path(self.plant,self.start,self.limits)
        np.testing.assert_array_equal(self.plant.data.qpos,before)
        np.testing.assert_allclose(path[-1],self.start)

    def test_out_of_limits_goal_rejected(self):
        goal=self.start.copy();goal[0]=self.limits[0,1]+.1
        with self.assertRaisesRegex(RuntimeError,'endpoint'):
            joint_path(self.plant,goal,self.limits)

    def test_deadline_and_cancellation_are_enforced(self):
        with self.assertRaisesRegex(RuntimeError,'deadline'):
            joint_path(self.plant,self.start,self.limits,timeout_s=-1)
        def cancelled():raise RuntimeError('Operator cancelled trial')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            joint_path(self.plant,self.start,self.limits,check=cancelled)

    def test_node_limit_bounds_obstructed_search(self):
        # A node-budget test needs clear endpoints, not a G1 start already
        # penetrating its hip. This analytic wall blocks only the direct edge.
        plant=NarrowObstacleTests().plant()
        goal=plant.data.qpos.copy();goal[0]=.1
        limits=np.array([[-.2,.2],[-.2,.2]]+[[-1e-5,1e-5]]*5)
        self.assertFalse(plant.pose_collision(plant.data))
        with self.assertRaisesRegex(RuntimeError,'node limit'):
            joint_path(plant,goal,limits,max_nodes=2)

class NarrowObstacleTests(unittest.TestCase):
    def plant(self):
        from types import SimpleNamespace
        # A 7 mm-wide collision interval lies between the old .04 joint samples.
        # The second joint lets the planner take a clear route around the wall.
        parts=['<mujoco><worldbody><geom name="wall" type="box" pos=".015 0 0" size=".003 .03 .1"/>']
        for index in range(7):
            axis='1 0 0' if index==0 else '0 1 0' if index==1 else '0 0 1'
            parts.append(f'<body><joint type="slide" axis="{axis}"/><inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>')
        parts.append('<geom name="probe" type="sphere" size=".0005"/>')
        parts.extend(['</body>']*7);parts.append('</worldbody></mujoco>')
        model=mujoco.MjModel.from_xml_string(''.join(parts));data=mujoco.MjData(model)
        data.qpos[0]=-.1;mujoco.mj_forward(model,data)
        def collision(state):return any(contact.dist<0 for contact in state.contact)
        return SimpleNamespace(model=model,data=data,qadr=np.arange(7),arm=np.arange(7),pose_collision=collision)

    def test_narrow_wall_is_avoided_by_tree_edges_and_shortcuts(self):
        plant=self.plant();start=plant.data.qpos.copy();goal=start.copy();goal[0]=.1
        limits=np.array([[-.2,.2],[-.2,.2]]+[[-1e-5,1e-5]]*5)
        # Establish the regression: the former tree sampling misses this wall.
        state=mujoco.MjData(plant.model)
        for alpha in np.linspace(0,1,6):
            state.qpos[:]=start+(goal-start)*alpha;mujoco.mj_forward(plant.model,state)
            self.assertFalse(plant.pose_collision(state))
        path,meta=joint_path(plant,goal,limits,timeout_s=5.)
        self.assertGreater(len(path),1)
        self.assertEqual(meta['edge_resolution_rad'],.004)
        previous=start
        # Independently check the returned route at eight times finer spacing.
        for waypoint in path:
            waypoint=np.asarray(waypoint)
            for alpha in np.linspace(0,1,int(np.ceil(np.max(np.abs(waypoint-previous))/.0005))+1):
                state.qpos[:]=previous+(waypoint-previous)*alpha
                mujoco.mj_forward(plant.model,state)
                self.assertFalse(plant.pose_collision(state))
            previous=waypoint
        np.testing.assert_allclose(previous,goal)
        np.testing.assert_array_equal(plant.data.qpos,start)


if __name__=='__main__':unittest.main()
