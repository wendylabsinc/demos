"""Audit physical motion, unsupported falls, and course obstacle collisions."""
import json
import numpy as np
import mujoco
from demo import Golf, ROOT, COURSES

results=[]
for course in COURSES:
    simulation=Golf(course)
    qpos=simulation.d.qpos.copy()
    qvel=simulation.d.qvel.copy()
    step=mujoco.mj_step
    try:
        mujoco.mj_step=lambda *_:None
        simulation.tick()
    finally:
        mujoco.mj_step=step
    assert np.array_equal(simulation.d.qpos,qpos), 'Controller directly changes positions'
    assert np.array_equal(simulation.d.qvel,qvel), 'Controller directly changes velocities'
    assert simulation.m.nmocap==0, 'Model has a kinematic mocap body'
    assert simulation.m.joint('floating_base_joint').type[0]==mujoco.mjtJoint.mjJNT_FREE
    assert not np.any(simulation.d.xfrc_applied) and not np.any(simulation.d.qfrc_applied)

    # Remove actuator support. Gravity must make the free-base robot collapse.
    simulation.m.actuator_gainprm[:]=0
    simulation.m.actuator_biasprm[:]=0
    for _ in range(1500): mujoco.mj_step(simulation.m,simulation.d)
    height=float(simulation.d.qpos[2])
    upright=float(simulation.d.xmat[simulation.m.body('pelvis').id,8])
    assert height<.5, 'Unpowered robot appears to have external support'
    result={'course':course,'no_direct_pose_writes':True,'no_direct_velocity_writes':True,'mocap_bodies':0,
            'free_base':True,'no_external_support_forces':True,
            'unpowered_base_height':height,'unpowered_upright':upright,'passed':True}

    # Initial conditions in an isolated test state, never in the running demo.
    test=mujoco.MjData(simulation.m)
    test.qpos[:]=simulation.pose
    if course=='bridge':
        test.qpos[simulation.ball:simulation.ball+3]=[1.8,.65,.036]
        for _ in range(400): mujoco.mj_step(simulation.m,test)
        result['ball_falls_through_bridge_gap']=bool(test.qpos[simulation.ball+2]<-.20)
        assert result['ball_falls_through_bridge_gap'], 'Bridge gap has a hidden floor'
    if course in ('gates','bridge'):
        test=mujoco.MjData(simulation.m); test.qpos[:]=simulation.pose
        rail=simulation.m.geom('curve_rail0_1')
        test.qpos[simulation.ball:simulation.ball+3]=[rail.pos[0],rail.pos[1]+.12,.036]
        test.qvel[simulation.bvel+1]=-1.
        contacts=0
        for _ in range(300):
            mujoco.mj_step(simulation.m,test)
            for contact in test.contact:
                names={simulation.m.geom(contact.geom1).name,simulation.m.geom(contact.geom2).name}
                contacts+=int('ball' in names and any(name and name.startswith('curve_rail') for name in names))
        result['ball_curve_rail_contact_steps']=contacts
        assert contacts>0, 'Curved rail does not physically collide with the ball'
    results.append(result)

(ROOT/'runtime').mkdir(exist_ok=True)
(ROOT/'runtime/physics-audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
