"""Measured-state bridge for the handoff IK and the existing arm SDK adapter.
This module has no DDS imports or publishers. Physics is used for FK/collision
checks only; simulation step(), contacts(), and can_position() are never used.
"""
import json
import os
import time
import threading
import numpy as np

MAX_MOTOR_TEMPERATURE_C = 75.0
# Conservative feedforward limits, below the model's actuator torque limits.
ARM_GRAVITY_LIMITS_NM = np.array([6., 6., 4., 4., 1.5, 1.5, 1.5] * 2)
ARM_GRAVITY_SLEW_NM_S = 5.0

SIM_TO_BRIDGE = [0, 1, 2, 5, 6, 3, 4]
BODY_NAMES = ([f'{s}_{j}_joint' for s in ('left','right') for j in
               ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
              + ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
              + [f'{s}_{j}_joint' for s in ('left','right') for j in
                 ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])


def rigid(value):
    m = np.asarray(value, dtype=float)
    if (m.shape != (4,4) or not np.all(np.isfinite(m)) or
        not np.allclose(m[3], [0,0,0,1], atol=1e-8) or
        not np.allclose(m[:3,:3].T@m[:3,:3], np.eye(3), atol=1e-5) or
        not np.isclose(np.linalg.det(m[:3,:3]), 1, atol=1e-5)):
        raise ValueError('Expected a proper rigid 4x4 calibration transform')
    return m


def calibration(value, camera_serial):
    if value.get('commissioned') is not True:
        raise ValueError('Camera/base calibration has not been accepted')
    if value.get('camera_serial') != camera_serial or value.get('device') != 'unitree-g1-nx-2.local':
        raise ValueError('Calibration identity does not match this G1 and camera')
    if value.get('calibration_method') == 'operator_accepted_factory_table':
        # Explicit alternative provenance. Plane fit error is NOT camera/base
        # position validation and must never be recorded as such.
        required = ('operator_accepted', 'factory_mount_position_confirmed', 'stationary_base_confirmed')
        if any(value.get(key) is not True for key in required) or value.get('independent_validation') is not False:
            raise ValueError('Factory/table estimate requires explicit operator acceptance and estimated provenance')
        if value.get('position_validation_rmse_m') is not None:
            raise ValueError('Factory/table estimate must not claim independent position validation')
        for key, maximum in [('table_plane_rmse_m', .003), ('tilt_correction_deg', 12.)]:
            v = value.get(key)
            if not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= maximum:
                raise ValueError(f'Invalid factory/table {key}')
        stamp = value.get('accepted_at_unix_s', float('nan'))
        if not np.isfinite(stamp) or not 0 <= time.time()-stamp <= 1800:
            raise ValueError('Model-based calibration acceptance expired; refresh scene and acceptance')
        if not value.get('source_frame_id') or not value.get('factory_mount_source') or not value.get('assumptions'):
            raise ValueError('Factory/table estimate lacks source evidence')
        for key, count in [('reference_base_q', 15), ('reference_imu_rpy', 3)]:
            v = np.asarray(value.get(key, []), dtype=float)
            if v.shape != (count,) or not np.isfinite(v).all():
                raise ValueError(f'Invalid model calibration {key}')
        center = np.asarray(value.get('table_center_xy_m'), dtype=float)
        half = np.asarray(value.get('table_half_size_xy_m'), dtype=float)
        near = value.get('table_near_edge_conservative_x_m', float('nan'))
        if (center.shape != (2,) or half.shape != (2,) or not np.isfinite(near)
                or np.any(half < 2.) or abs(center[1]) > .001
                or abs(center[0]-half[0]-near) > .001):
            raise ValueError('Model calibration must conservatively extend table beyond arm workspace')
    else:
        rmse = value.get('position_validation_rmse_m')
        if not value.get('measurement_evidence') or not isinstance(rmse, (int, float)) or not np.isfinite(rmse) or not 0 <= rmse <= .01:
            raise ValueError('Calibration needs independent <=10 mm position validation evidence')
    rigid(value['camera_to_world']); rigid(value['pelvis_to_world'])
    for field in ('table_height_m', 'can_radius_m', 'can_height_m'):
        if not np.isfinite(value[field]) or value[field] <= 0:
            raise ValueError(f'Expected finite positive {field}')
    for field in ('table_center_xy_m', 'table_half_size_xy_m'):
        v = np.asarray(value[field], dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all() or (field == 'table_half_size_xy_m' and np.any(v <= 0)):
            raise ValueError(f'Invalid measured {field}')
    if not .645 <= value['table_height_m'] <= .955:
        raise ValueError('Table is outside upstream handoff workspace')
    # Never silently assume the simulation cylinder matches the physical can.
    if abs(value['can_radius_m']-.033) > .002 or abs(value['can_height_m']-.122) > .008:
        raise ValueError('Measured can dimensions require updating the upstream model/grasp')
    if value.get('operator_with_stop') is not True or value.get('support') not in ('factory_balance','harness'):
        raise ValueError('Operator stop and support setup must be recorded')
    return value


def calibration_start_guard(config, snapshot):
    """Reject a model-based fixed transform after base/waist posture changes."""
    if config.get('calibration_method') != 'operator_accepted_factory_table':
        return
    body = snapshot_guard(snapshot)['body']
    if np.max(np.abs(np.asarray(body['q'][:15])-config['reference_base_q'])) > .04:
        raise ValueError('Base/waist posture changed since model calibration acceptance')
    delta = np.asarray(body['imu_rpy'])-config['reference_imu_rpy']
    delta = (delta+np.pi) % (2*np.pi)-np.pi
    if np.max(np.abs(delta)) > .025:
        raise ValueError('Base attitude changed since model calibration acceptance')


def calibrated_scene(config, x=.35, y=.18, decorate=None):
    """Compile configured base/table estimates, including collision bounds."""
    import mujoco
    from g1_can_handoff.sim import scene
    try:
        clearance=float(os.environ.get('HANDOFF_TABLE_CLEARANCE_M','0.01'))
        if not np.isfinite(clearance) or not .005<=clearance<=.03:raise ValueError()
    except ValueError as exc:
        raise ValueError('Table clearance must be finite and between 0.005 and 0.03 m') from exc
    root_pose = rigid(config['pelvis_to_world'])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, root_pose[:3, :3].ravel())
    def configure(root):
        pelvis = root.find(".//body[@name='pelvis']")
        pelvis.set('pos', ' '.join(map(str, root_pose[:3, 3])))
        pelvis.set('quat', ' '.join(map(str, quat)))
        table = root.find(".//geom[@name='table']")
        table.set('pos', ' '.join(map(str, [*config['table_center_xy_m'], config['table_height_m']-.025])))
        table.set('size', ' '.join(map(str, [*config['table_half_size_xy_m'], .025])))
        # Generate proximity contacts without moving the measured table plane.
        table.set('margin',str(clearance));table.set('gap',str(clearance))
        if decorate is not None:
            decorate(root)
    plant=scene(height=config['table_height_m'], x=x, y=y, decorate=configure)
    plant.table_clearance_m=clearance
    return plant


def sim_hand_to_bridge(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected seven finite finger targets')
    return q[SIM_TO_BRIDGE]


def snapshot_guard(snapshot, now=None):
    now = time.monotonic() if now is None else now
    for name, count in [('body',29), ('left',7), ('right',7)]:
        sample = snapshot[name]
        if not 0 <= now-sample['time'] <= .2:
            raise ValueError(f'Stale {name} measured feedback')
        q = np.asarray(sample['q'], dtype=float)
        if q.ndim != 1 or len(q) != count or not np.all(np.isfinite(q)):
            raise ValueError(f'Invalid {name} measured feedback')
        temperature = sample.get('temperature_max', float('nan'))
        if not np.isfinite(temperature) or temperature >= MAX_MOTOR_TEMPERATURE_C:
            raise ValueError(f'{name} temperature {temperature} C reaches the {MAX_MOTOR_TEMPERATURE_C:g} C guard or is invalid')
        if 'faults' not in sample or any(sample['faults']):
            raise ValueError(f'{name} motor fault or missing fault feedback')
        for field in ('dq', 'tau_est'):
            values = np.asarray(sample.get(field, []), dtype=float)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {name} {field} feedback')
    rpy = np.asarray(snapshot['body'].get('imu_rpy', []), dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError('Missing or invalid body attitude feedback')
    return snapshot


class PhysicalPlant:
    def __init__(self, model_plant, adapter, hand, read, config, event, clock=time, cancel=None):
        self.command_lock = threading.RLock()
        self.clock = clock
        self.cancel = cancel
        self.perception_guard = None
        self.ik = model_plant
        self.adapter, self.sender, self.read = adapter, hand, read
        self.config, self.event = config, event
        limits=np.asarray(config['arm_limits_rad'],dtype=float)
        if limits.shape!=(14,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>=limits[:,1]):
            raise ValueError('Expected fourteen measured arm limit pairs')
        # Solve within the actual hardware range instead of solving with
        # simulation limits and rejecting the chosen branch afterward.
        self.ik.model.jnt_range[self.ik.joints[self.ik.arm]]=limits[:7]
        snap = snapshot_guard(read(), now=self.clock.monotonic())
        self.base_joints = np.array(snap['body']['q'][:15])
        self.base_rpy = np.array(snap['body'].get('imu_rpy',[0,0,0]))
        self.right_arm = np.array(snap['body']['q'][22:29])
        self.left_arm = np.array(snap['body']['q'][15:22])
        self.right_hand = np.array(snap['right']['q'])[SIM_TO_BRIDGE]
        self.left_hand = np.array(snap['left']['q'])
        self.cancelled = False
        self.contact_evidence = None
        self.staging_waypoints = []
        self.staging_start = None
        self.path_preflight_ok = False
        self.sync(snap)
        import mujoco
        model=self.ik.model
        torso=model.body('torso_link').id
        elbow=model.body('left_elbow_link').id
        self.elbow_torso_pairs=[(a,b) for a in range(model.ngeom) for b in range(model.ngeom)
            if model.geom_bodyid[a]==torso and model.geom_bodyid[b]==elbow
            and model.geom_contype[a] and model.geom_contype[b]]
        self.table_pairs=[]
        if getattr(self.ik,'table_clearance_m',0.)>0:
            table=model.geom('table').id
            self.table_pairs=[(table,g) for g in range(model.ngeom) if model.geom_contype[g]
                              and model.body(model.geom_bodyid[g]).name.startswith('left_')]
        self.gravity_data = mujoco.MjData(self.ik.model)
        self.arm_dofs = np.array([self.ik.model.jnt_dofadr[self.ik.model.joint(n).id]
                                  for n in BODY_NAMES[15:]])
        self.arm_tau_command = np.zeros(14)
        self.arm_tau_time = self.clock.monotonic()

    def sync(self, snap):
        import mujoco
        if self.config.get('calibration_method')=='operator_accepted_factory_table':
            from pose_alignment import corrected_config
            body=snap['body']
            corrected=corrected_config(self.config,{'timestamp':body['time'],
                'rpy':body['imu_rpy'],'waist_q':body['q'][12:15]})
            root=np.asarray(corrected['pelvis_to_world'])
            pelvis=self.ik.model.body('pelvis').id
            self.ik.model.body_pos[pelvis]=root[:3,3]
            mujoco.mju_mat2Quat(self.ik.model.body_quat[pelvis],root[:3,:3].ravel())
        for name,q in zip(BODY_NAMES, snap['body']['q'][:29]):
            self.ik.data.qpos[self.ik.model.jnt_qposadr[self.ik.model.joint(name).id]]=q
        for side,values in [('left', snap['left']['q']), ('right', np.array(snap['right']['q'])[SIM_TO_BRIDGE])]:
            # Model uses thumb,index,middle on both hands.
            sim_order = np.asarray(values)[SIM_TO_BRIDGE]
            for suffix,q in zip(('thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1'),sim_order):
                jid=self.ik.model.joint(f'{side}_hand_{suffix}_joint').id
                self.ik.data.qpos[self.ik.model.jnt_qposadr[jid]]=q
        mujoco.mj_forward(self.ik.model,self.ik.data)

    def observe_can(self, observation):
        """Place only the FK collision proxy at the camera-measured can pose."""
        import mujoco
        center=np.asarray(observation.center,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise ValueError('Expected a finite observed can center')
        adr=self.ik.model.jnt_qposadr[self.ik.model.joint('can_free').id]
        self.ik.data.qpos[adr:adr+3]=center
        self.ik.data.qpos[adr+3:adr+7]=[1,0,0,0]
        mujoco.mj_forward(self.ik.model,self.ik.data)
        self.event('observed_can_collision_proxy',center_m=center.tolist(),source='camera')

    def set_contact_phase(self, phase):
        self.ik.set_contact_phase(phase)

    def solve(self, position, rotation=None):
        self.sync(self.check())
        collision=self.ik.collision_details(self.ik.data)
        if collision is not None:
            self.event("starting_pose_collision",collision=collision)
            raise RuntimeError(f"Starting measured pose intersects robot/table: {collision}")
        return self._pose_ik(position, rotation)

    def _robot_contact_pairs(self, data):
        model=self.ik.model
        pairs=set()
        for contact in data.contact:
            names=[model.body(model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if ('can' in names or 'world' in names or
                    not any(n.startswith('left_') for n in names) or
                    all('left_hand' in n or n=='left_wrist_yaw_link' for n in names)):
                continue
            pairs.add(tuple(sorted((int(contact.geom1),int(contact.geom2)))))
        return sorted(pairs)

    def _pose_ik(self, position, rotation=None):
        """Keep existing endpoint IK; recover only a table/torso rejection."""
        import mujoco
        from scipy.optimize import least_squares
        try:
            return self.ik.solve(position,rotation)
        except ValueError as exc:
            if not str(exc).startswith('Target pose collision:'):
                raise
            collision=self.ik.collision_details(self.ik.ik_data)
            if (collision is None or
                    not ('table' in collision['geoms'] or
                         set(collision['bodies'])=={'torso_link','left_elbow_link'} or
                         all(n.startswith('left_') for n in collision['bodies']))):
                raise
            original_error=exc
        model,data=self.ik.model,self.ik.ik_data
        seed=data.qpos[self.ik.qadr[self.ik.arm]].copy()
        robot_pairs=self._robot_contact_pairs(data)
        data.qpos[:]=self.ik.data.qpos
        adr=self.ik.qadr[self.ik.arm]
        start=data.qpos[adr].copy()
        limits=np.asarray(self.config['arm_limits_rad'])[:7]
        lo,hi=limits[:,0]+.025,limits[:,1]-.025
        rotation=np.eye(3) if rotation is None else rotation
        table_gap=getattr(self.ik,'table_clearance_m',0.)+.001
        def error(q):
            self.check()
            data.qpos[adr]=q
            mujoco.mj_forward(model,data)
            pose=np.r_[data.site_xpos[self.ik.site]-position,
                       .15*(data.site_xmat[self.ik.site].reshape(3,3)-rotation).ravel()]
            torso=self._geometry_gaps(data,self.elbow_torso_pairs)
            table=self._geometry_gaps(data,self.table_pairs)
            robot=self._geometry_gaps(data,robot_pairs)
            return np.r_[pose,5*np.maximum(0,.003-torso),
                         10*np.maximum(0,table_gap-table),5*np.maximum(0,.003-robot),.003*(q-start)]
        result=least_squares(error,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=250)
        error(result.x)
        if (np.linalg.norm(result.fun[:3])>.008 or np.linalg.norm(result.fun[3:12])>.02
                or self.ik.collision_details(data) is not None):
            raise original_error
        self.event('endpoint_clearance_ik_recovered',phase_name=self.ik.contact_phase,
                   original_collision=collision,position_error_m=float(np.linalg.norm(result.fun[:3])))
        return result.x

    def _geometry_gaps(self, data, pairs):
        import mujoco
        return np.array([mujoco.mj_geomDistance(self.ik.model,data,a,b,.1,None)
                         for a,b in pairs])

    def check(self):
        if self.cancelled or (self.cancel is not None and self.cancel.is_set()):
            raise RuntimeError('Operator cancelled trial')
        if self.perception_guard is not None:self.perception_guard()
        snap=snapshot_guard(self.read(), now=self.clock.monotonic())
        if np.max(np.abs(np.array(snap['body']['q'][:15])-self.base_joints))>.06:
            raise RuntimeError('Base/waist posture moved beyond fixed-frame calibration tolerance')
        if np.max(np.abs(np.array(snap['body'].get('imu_rpy',[0,0,0]))-self.base_rpy))>.04:
            raise RuntimeError('Base attitude moved beyond fixed-frame calibration tolerance')
        if np.max(np.abs(np.array(snap['body']['q'][22:29])-self.right_arm))>.12:
            raise RuntimeError('Right arm deviated from measured hold pose')
        if self.sender is not None and hasattr(self.sender, 'feedback_temperatures'):
            # Runtime samples preserve raw SDK ordering; sender expects bridge
            # ordering. Reuse fresh readings instead of DDS reads on this path.
            left = np.max(np.asarray(snap['left']['temperatures'], dtype=float), axis=1)
            right = np.max(np.asarray(snap['right']['temperatures'], dtype=float), axis=1)
            self.sender.feedback_temperatures(left.tolist(), right[SIM_TO_BRIDGE].tolist())
        return snap

    def _move_poses(self, position, duration=2., rotation=None):
        if self.ik.contact_phase == 'lift':
            return self._cartesian_lift_poses(position,duration,rotation)
        if self.ik.contact_phase in ('align','reach'):
            return self._cartesian_transfer_poses(position,duration,rotation)
        return self._joint_poses(self._pose_ik(position,rotation),duration)

    def _cartesian_transfer_poses(self,position,duration=2.,rotation=None):
        if self.ik.contact_phase not in ('align','reach'):
            raise RuntimeError('Cartesian transfer requires align or reach phase')
        return self._cartesian_lift_poses(position,duration,rotation,allow_descent=True)

    def _cartesian_lift_poses(self, position, duration=2., rotation=None, *, allow_descent=False):
        """Lift along the Cartesian line, checking every connecting joint path."""
        import mujoco
        original=self.ik.data.qpos.copy()
        start=self.ik.data.site_xpos[self.ik.site].copy()
        target=np.asarray(position,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():
            raise ValueError('Expected finite Cartesian lift target')
        distance=float(np.linalg.norm(target-start))
        if allow_descent and self.ik.contact_phase not in ('align','reach'):
            raise RuntimeError('Cartesian descent requires align or reach phase')
        if distance>.20 or (not allow_descent and target[2]<start[2]-.005):
            raise RuntimeError('Cartesian lift must ascend by at most 0.20 m')
        count=max(1,int(np.ceil(distance/.01)))
        poses=[]
        pending=[(start+(target-start)*(index/count),duration/count,0)
                 for index in range(count,0,-1)]
        attempts=0
        completed=0
        try:
            while pending:
                self.check()
                waypoint,segment_duration,depth=pending.pop()
                attempts+=1
                if attempts>100:
                    raise RuntimeError('Continuous Cartesian lift IK cannot preserve the grasp pose: subdivision budget exhausted')
                try:
                    goal=self._continuous_lift_ik(waypoint,rotation)
                except RuntimeError as exc:
                    if not str(exc).startswith('Continuous Cartesian lift IK cannot preserve'):
                        raise
                    current=self.ik.data.site_xpos[self.ik.site].copy()
                    if depth>=4 or np.linalg.norm(waypoint-current)<=.001:
                        raise
                    # Retry a shorter step from the last accepted IK state.
                    # Collision, telemetry and cancellation errors never trigger retries.
                    pending.append((waypoint,segment_duration/2,depth+1))
                    pending.append(((current+waypoint)/2,segment_duration/2,depth+1))
                    continue
                goal,segment=self._joint_poses(goal,segment_duration)
                poses.extend(segment)
                self.ik.data.qpos[self.ik.qadr[self.ik.arm]]=goal
                mujoco.mj_forward(self.ik.model,self.ik.data)
                completed+=1
            self.event('cartesian_transfer_planned' if allow_descent else 'cartesian_lift_planned',
                       phase_name=self.ik.contact_phase,waypoints=completed,ik_attempts=attempts,
                       start=start.tolist(),target=target.tolist())
            return goal,poses
        finally:
            self.ik.data.qpos[:]=original
            mujoco.mj_forward(self.ik.model,self.ik.data)

    def _continuous_lift_ik(self, position, rotation):
        """Continue the same branch; recover table clearance only when needed."""
        import mujoco
        from scipy.optimize import least_squares
        model,data=self.ik.model,self.ik.ik_data
        data.qpos[:]=self.ik.data.qpos
        adr=self.ik.qadr[self.ik.arm]
        start=data.qpos[adr].copy()
        limits=np.asarray(self.config['arm_limits_rad'])[:7]
        lo=np.maximum(limits[:,0],start-.2)
        hi=np.minimum(limits[:,1],start+.2)
        rotation=np.eye(3) if rotation is None else rotation
        table_gap=getattr(self.ik,'table_clearance_m',0.)+.001
        fractions=(.25,.5,.75,1.)
        def path_table_gaps(q,pairs):
            gaps=[]
            for alpha in fractions:
                data.qpos[adr]=start+alpha*(q-start)
                mujoco.mj_forward(model,data)
                gaps.append(self._geometry_gaps(data,pairs))
            return np.asarray(gaps)
        def error(q,avoid_torso=False,table_pairs=(),robot_pairs=()):
            if table_pairs or robot_pairs:self.check()
            data.qpos[adr]=q
            mujoco.mj_forward(model,data)
            pose=np.r_[data.site_xpos[self.ik.site]-position,
                       .15*(data.site_xmat[self.ik.site].reshape(3,3)-rotation).ravel()]
            if not avoid_torso:return pose
            gaps=np.array([mujoco.mj_geomDistance(model,data,a,b,.1,None)
                           for a,b in self.elbow_torso_pairs])
            # Pose-only IK can curl the redundant elbow into the torso during
            # a valid Cartesian reach/lift. Bias it away before contact, and
            # damp null-space drift. Final joint-path collision checks still
            # decide acceptance; these residuals grant no collision exemption.
            table=path_table_gaps(q,table_pairs) if table_pairs else np.array([])
            robot=path_table_gaps(q,robot_pairs) if robot_pairs else np.array([])
            return np.r_[pose,5*np.maximum(0,.003-gaps),
                         10*np.maximum(0,table_gap-table).ravel(),
                         5*np.maximum(0,.003-robot).ravel(),.003*(q-start)]
        result=least_squares(error,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=250)
        error(result.x)
        gaps=[mujoco.mj_geomDistance(model,data,a,b,.1,None) for a,b in self.elbow_torso_pairs]
        if any(gap<.003 for gap in gaps):
            result=least_squares(lambda q:error(q,True),np.clip(start,lo,hi),
                                 bounds=(lo,hi),max_nfev=250)
            error(result.x)
        robot_pairs=set()
        for alpha in fractions:
            data.qpos[adr]=start+alpha*(result.x-start)
            mujoco.mj_forward(model,data)
            robot_pairs.update(self._robot_contact_pairs(data))
        robot_pairs=sorted(robot_pairs)
        active=[]
        refine_table=False
        if self.table_pairs:
            table=path_table_gaps(result.x,self.table_pairs)
            active=[pair for pair,gap in zip(self.table_pairs,np.min(table,axis=0))
                    if gap<table_gap+.005]
            refine_table=bool(np.any(table<table_gap))
            before_gap=float(np.min(table))
        if refine_table or robot_pairs:
            result=least_squares(lambda q:error(q,True,active,robot_pairs),np.clip(result.x,lo,hi),
                                 bounds=(lo,hi),max_nfev=250)
            if refine_table:
                table=path_table_gaps(result.x,self.table_pairs)
                self.event('table_clearance_ik_refined',phase_name=self.ik.contact_phase,
                           before_distance_m=before_gap,after_distance_m=float(np.min(table)),
                           required_clearance_m=self.ik.table_clearance_m)
            if robot_pairs:
                self.event('robot_clearance_ik_refined',phase_name=self.ik.contact_phase,
                           geom_pairs=[list(pair) for pair in robot_pairs],
                           after_distance_m=float(np.min(path_table_gaps(result.x,robot_pairs))))
        error(result.x)
        if np.linalg.norm(result.fun[:3])>.008 or np.linalg.norm(result.fun[3:12])>.02:
            self.event('continuous_lift_ik_failed',position_error_m=float(np.linalg.norm(result.fun[:3])),
                       rotation_error=float(np.linalg.norm(result.fun[3:12])),target_position=np.asarray(position).tolist(),
                       start_q=start.tolist(),solution_q=result.x.tolist(),
                       local_bound_active=bool(np.any(np.abs(result.x-start)>.199)))
            raise RuntimeError('Continuous Cartesian lift IK cannot preserve the grasp pose')
        return result.x

    def _joint_poses(self, goal, duration=2.):
        import mujoco
        start=self.ik.data.qpos[self.ik.qadr[self.ik.arm]].copy()
        goal=np.asarray(goal,dtype=float)
        if goal.shape!=(7,) or not np.isfinite(goal).all():
            raise ValueError('Expected seven finite staging joint targets')
        limits=np.asarray(self.config['arm_limits_rad'])[:7]
        if np.any(goal<limits[:,0]) or np.any(goal>limits[:,1]):
            raise RuntimeError('IK target exceeds commissioned physical arm limits')
        # Smoothstep peak speed = 1.5 * joint travel / duration.
        travel=float(np.max(np.abs(goal-start)))
        duration=max(duration,1.5*travel/.25,np.sqrt(6*travel/.5))
        poses=[start+(goal-start)*(t*t*(3-2*t)) for t in np.linspace(0,1,max(2,int(duration/.02)))]
        self.ik.ik_data.qpos[:]=self.ik.data.qpos
        for pose_index,q in enumerate(poses):
            self.ik.ik_data.qpos[self.ik.qadr[self.ik.arm]]=q
            mujoco.mj_forward(self.ik.model,self.ik.ik_data)
            details=self.ik.collision_details(self.ik.ik_data)
            if details is not None:
                self.event('trajectory_collision',phase_name=self.ik.contact_phase,collision=details,joint_q=q.tolist(),path_fraction=pose_index/(len(poses)-1))
                raise RuntimeError(f'Interpolated physical trajectory intersects robot/table: {details}')
        return goal, poses

    def preflight(self, points, rotation):
        """Stage with measured fingers, then check open/reach/close/lift paths."""
        import mujoco
        from g1_can_handoff.task import OPEN, CLOSED
        self.sync(self.check())
        original=self.ik.data.qpos.copy()
        self.staging_start=original[self.ik.qadr[self.ik.arm]].copy()
        self.staging_waypoints=[]
        self.path_preflight_ok=False
        staging_goal=None
        def fingers(target):
            start=self.ik.data.qpos[self.ik.qadr[self.ik.fingers]].copy()
            for alpha in np.linspace(0,1,51):
                self.ik.data.qpos[self.ik.qadr[self.ik.fingers]]=start+alpha*(np.asarray(target)-start)
                mujoco.mj_forward(self.ik.model,self.ik.data)
                details=self.ik.collision_details(self.ik.data)
                if details is not None:
                    self.event('finger_trajectory_collision',phase_name=self.ik.contact_phase,collision=details)
                    raise RuntimeError(f'Planned finger trajectory intersects robot/table: {details}')
        try:
            self.set_contact_phase('pregrasp')
            collision=self.ik.collision_details(self.ik.data)
            if collision is not None:
                self.event('starting_pose_collision',collision=collision)
                raise RuntimeError(f'Starting measured pose intersects robot/table: {collision}')
            for index,position in enumerate(points[:4]):
                self.set_contact_phase(('approach','align','reach','lift')[index])
                if index==3:fingers(CLOSED)
                goal=None if index in (1,2,3) else self._pose_ik(position,rotation)
                try:
                    if index==3:goal,_=self._cartesian_lift_poses(position,rotation=rotation)
                    elif index in (1,2):goal,_=self._cartesian_transfer_poses(position,rotation=rotation)
                    else:self._joint_poses(goal)
                except RuntimeError as exc:
                    if (index!=0 or not self.config.get('staging_enabled',True) or
                            'trajectory intersects' not in str(exc)):
                        raise
                    # Check reach, close and lift before spending the staging
                    # search budget on a grasp that may already be infeasible.
                    staging_goal=goal.copy()
                self.ik.data.qpos[self.ik.qadr[self.ik.arm]]=goal
                mujoco.mj_forward(self.ik.model,self.ik.data)
                if index==0:fingers(OPEN)
            if staging_goal is not None:
                self.ik.data.qpos[:]=original
                self.set_contact_phase('staging')
                mujoco.mj_forward(self.ik.model,self.ik.data)
                from staging import joint_path
                path,metadata=joint_path(self.ik,staging_goal,np.asarray(self.config['arm_limits_rad'])[:7],
                                         check=self.check)
                for waypoint in path:
                    self._joint_poses(waypoint)
                    self.ik.data.qpos[self.ik.qadr[self.ik.arm]]=waypoint
                    mujoco.mj_forward(self.ik.model,self.ik.data)
                self.staging_waypoints=path
                self.event('staging_planned',start_q=self.staging_start.tolist(),
                           joint_waypoints=path,**metadata)
            self.path_preflight_ok=True
            self.event('path_preflight_passed',waypoints_checked=4,hand_paths_checked=2,motion_commands_sent=0,
                       table_clearance_m=getattr(self.ik,'table_clearance_m',0.))
        finally:
            self.set_contact_phase('pregrasp')
            self.ik.data.qpos[:]=original
            mujoco.mj_forward(self.ik.model,self.ik.data)

    def validate_start(self):
        snap=self.check()
        if not self.path_preflight_ok or self.staging_start is None:
            raise RuntimeError('Pickup path has not been preflighted')
        if np.max(np.abs(np.asarray(snap['body']['q'][15:22])-self.staging_start))>.04:
            self.event('start_pose_changed',planned_q=self.staging_start.tolist(),
                       measured_q=snap['body']['q'][15:22],
                       error_rad=(np.asarray(snap['body']['q'][15:22])-self.staging_start).tolist())
            raise RuntimeError('Measured starting arm posture changed after planning; replan required')
        return snap

    def stage(self):
        self.set_contact_phase('staging')
        self.sync(self.validate_start())
        for index,waypoint in enumerate(self.staging_waypoints):
            self.sync(self.check())
            goal,poses=self._joint_poses(waypoint)
            self.event('staging_segment',index=index,goal_q=goal.tolist())
            self._execute_poses(goal,poses)

    def move(self, position, duration=2., rotation=None):
        snap=self.check(); self.sync(snap)
        goal,poses=self._move_poses(position,duration,rotation)
        self._execute_poses(goal,poses)
        if np.linalg.norm(self.ik.data.site_xpos[self.ik.site]-position)>.025:
            raise RuntimeError('Measured FK failed the 25 mm target tolerance')

    def _execute_poses(self, goal, poses):
        for q in poses:
            started=self.clock.monotonic()
            with self.command_lock:
                snap=self.check()
                error=np.max(np.abs(np.asarray(snap['body']['q'][15:22])-q))
                if error>.15: raise RuntimeError('Measured arm tracking error exceeds 0.15 rad')
                self.command_arm(q,snap)
                self.sender.ctrl_dual_hand(self.left_hand,self.right_hand)
                self.event('target',arm=np.r_[q,self.right_arm].tolist(),tracking_error_rad=float(error),
                           gravity_feedforward_nm=self.arm_tau_command.tolist())
            self.clock.sleep(max(0,.02-(self.clock.monotonic()-started)))
        deadline=self.clock.monotonic()+2
        while self.clock.monotonic()<deadline:
            with self.command_lock:
                snap=self.check()
                self.command_arm(goal,snap)
                if np.max(np.abs(np.asarray(snap['body']['q'][15:22])-goal))<.04: break
            self.clock.sleep(.02)
        else:
            delta=np.asarray(snap['body']['q'][15:22])-goal
            self.event('arm_settle_failed',goal_q=goal.tolist(),measured_q=snap['body']['q'][15:22],
                       error_rad=delta.tolist(),gravity_feedforward_nm=self.arm_tau_command.tolist())
            raise RuntimeError('Physical arm did not settle at the IK target')
        self.sync(snap)

    def command_arm(self, left_q, snapshot):
        """SDK second argument is feedforward torque, not joint velocity."""
        import mujoco
        self.sync(snapshot)
        self.check_live_geometry(left_q=left_q,hand_q=self.left_hand)
        self.gravity_data.qpos[:] = self.ik.data.qpos
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.ik.model, self.gravity_data)
        target = self.gravity_data.qfrc_bias[self.arm_dofs].copy()
        if not np.isfinite(target).all():
            raise RuntimeError('Invalid arm gravity compensation')
        if not self.config.get('arm_gravity_compensation', True):
            target[:] = 0
        target = np.clip(target, -ARM_GRAVITY_LIMITS_NM, ARM_GRAVITY_LIMITS_NM)
        now = self.clock.monotonic()
        # No jump after planning or a long gap between calls.
        dt = min(.02, max(0., now-self.arm_tau_time))
        change = ARM_GRAVITY_SLEW_NM_S*dt
        self.arm_tau_command += np.clip(target-self.arm_tau_command, -change, change)
        self.arm_tau_time = now
        self.adapter.ctrl_dual_arm(np.r_[left_q, self.right_arm], self.arm_tau_command.copy())
        self.left_arm=np.asarray(left_q,dtype=float).copy()

    def hold_arm(self,snapshot):
        if np.max(np.abs(np.asarray(snapshot['body']['q'][15:22])-self.left_arm))>.15:
            raise RuntimeError('Measured arm tracking error exceeds 0.15 rad')
        self.command_arm(self.left_arm,snapshot)

    def settle_control(self,archive_guard=lambda:None):
        """Hold the bounded SDK reference until measured control is stable."""
        from control_settle import wait_for_stable_control
        def sample():
            with self.command_lock:
                snapshot=self.check();self.hold_arm(snapshot)
                return snapshot
        return wait_for_stable_control(sample,self.cancel or threading.Event(),self.event,archive_guard,self.clock)

    def inherit_control_hold(self,previous):
        """A new calibrated planning model must not restart the torque ramp."""
        if previous.cancelled or self.adapter is not previous.adapter or self.sender is not previous.sender or self.clock is not previous.clock:
            raise RuntimeError('Cannot inherit a hold from a different or stopped controller')
        self.arm_tau_command=previous.arm_tau_command.copy()
        self.arm_tau_time=previous.arm_tau_time
        self.left_arm=previous.left_arm.copy();self.right_arm=previous.right_arm.copy()

    def check_live_geometry(self,left_q=None,hand_q=None):
        """Recheck commands against the current measured base attitude."""
        if (getattr(self.ik,'table_clearance_m',0.)<=0 and
                self.config.get('calibration_method')!='operator_accepted_factory_table'):return
        import mujoco
        data=self.gravity_data;data.qpos[:]=self.ik.data.qpos
        if left_q is not None:
            data.qpos[self.ik.qadr[self.ik.arm]]=left_q
            for name,q in zip(BODY_NAMES[22:29],self.right_arm):
                data.qpos[self.ik.model.jnt_qposadr[self.ik.model.joint(name).id]]=q
        if hand_q is not None:
            for side,values in [('left',hand_q),('right',self.right_hand)]:
                for suffix,q in zip(('thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1'),
                                    np.asarray(values)[SIM_TO_BRIDGE]):
                    data.qpos[self.ik.model.jnt_qposadr[self.ik.model.joint(side+'_hand_'+suffix+'_joint').id]]=q
        mujoco.mj_forward(self.ik.model,data)
        collision=self.ik.collision_details(data)
        if collision:raise RuntimeError('Live target intersects at measured base attitude: '+str(collision))

    def hand(self, goal, duration=1.):
        end=sim_hand_to_bridge(goal);start=self.left_hand.copy()
        for t in np.linspace(0,1,max(2,int(duration/.02))):
            with self.command_lock:
                snap=self.check();self.sync(snap);self.left_hand=start+t*(end-start)
                self.hold_arm(snap)
                self.check_live_geometry(hand_q=self.left_hand)
                self.sender.ctrl_dual_hand(self.left_hand,self.right_hand)
                self.event('hand_target',left=self.left_hand.tolist(),right=self.right_hand.tolist())
            self.clock.sleep(.02)

    def contacts(self):
        # Evidence is an obstruction/torque proxy, not simulated contact or proof
        # of object identity. Independent camera lift must establish grasp.
        snap=self.check();sample=snap['left'];q=np.asarray(sample['q']);tau=np.asarray(sample['tau_est'])
        groups=[]
        for name,indices in [('thumb',[1,2]),('middle',[3,4]),('index',[5,6])]:
            if any(abs(q[i]-self.left_hand[i])>.12 and abs(sample["dq"][i])<.1 for i in indices): groups.append(name)
        self.contact_evidence={'kind':'stationary_position_obstruction_proxy_unverified_contact','groups':groups,
                               'q':q.tolist(),'tau_est':tau.tolist()}
        self.event('contact_proxy',**self.contact_evidence)
        return groups

    def step(self, seconds):
        until=self.clock.monotonic()+seconds
        while self.clock.monotonic()<until:
            self.maintain_hand()
            self.clock.sleep(.02)

    def maintain_hand(self):
        with self.command_lock:
            snapshot=self.check();self.hold_arm(snapshot)
            self.sender.ctrl_dual_hand(self.left_hand,self.right_hand)

    def freeze(self):
        self.cancelled=True
        with self.command_lock:
            if self.adapter is not None:
                self.adapter.hold_arm_measured()
            self.event('frozen',automatic_home=False,hand_release=False)
