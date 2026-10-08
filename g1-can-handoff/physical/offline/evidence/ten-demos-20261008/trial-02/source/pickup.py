"""Stationary pickup sequence shared by physical and simulated G1 backends.

Perception and plant feedback are supplied by the backend. No object pose or
simulation contacts bypass the measured-state or camera verification gates.
"""
import numpy as np
from g1_can_handoff.task import plan, OPEN, CLOSED


def prepare(plant, observe, event):
    observation, baseline = observe('planning', True)
    plant.observe_can(observation)
    rejected=[]
    def validate_candidate(points,rotation):
        try:
            plant.preflight(points,rotation)
            return True
        except (RuntimeError,ValueError) as exc:
            # Only geometric/search failures permit another grasp candidate.
            # Telemetry, stop, temperature and posture errors abort immediately.
            geometric=('Interpolated physical trajectory intersects',
                       'Planned finger trajectory intersects',
                       'Continuous Cartesian lift IK',
                       'Target pose collision',
                       'Target cannot be reached',
                       'Staging endpoint collides',
                       'Staging planning deadline reached',
                       'Staging planning node limit reached',
                       'No staging path within bounded search')
            if not str(exc).startswith(geometric):raise
            reason=str(exc).split(':',1)[0]
            if reason not in rejected:rejected.append(reason)
            event('grasp_candidate_rejected',rotation=rotation.tolist(),reason=str(exc))
            return False
    try:points, rotation = plan(plant, observation,validate=validate_candidate,include_offer=False)
    except ValueError as exc:
        if rejected:raise ValueError(str(exc)+'; preflight rejected: '+'; '.join(rejected)) from exc
        raise
    refreshed, track = observe('plan-refresh', True)
    if (np.linalg.norm(refreshed.center-observation.center) > .01 or
            abs(refreshed.table_height-observation.table_height) > .005):
        raise RuntimeError('Can or table moved during planning')
    event('plan', waypoints=[p.tolist() for p in points],
          rotation=rotation.tolist(), observation=track)
    return observation, baseline, points, rotation


def execute(plant, prepared, observe, event):
    observation, baseline, points, rotation = prepared
    plant.validate_start()
    plant.set_contact_phase('pregrasp')
    plant.hand(OPEN, .8)
    plant.stage()
    for phase, point in zip(('approach', 'align', 'reach'), points[:3]):
        fresh, vision = observe(phase, True)
        if (np.linalg.norm(fresh.center-observation.center) > .01 or
                abs(fresh.table_height-observation.table_height) > .005):
            raise RuntimeError('Can or table moved during approach; replanning required')
        plant.observe_can(fresh)
        plant.set_contact_phase(phase)
        event(phase)
        plant.move(point, rotation=rotation)
    fresh, vision = observe('pre-close', True)
    if (np.linalg.norm(fresh.center-observation.center) > .01 or
            abs(fresh.table_height-observation.table_height) > .005):
        raise RuntimeError('Can or table moved during final reach; refusing to close')
    plant.observe_can(fresh)
    plant.set_contact_phase('close')
    event('close')
    plant.hand(CLOSED, 1.2)
    plant.step(.5)
    # Closing contact can still be settling after the nominal half second.
    # Require the same two-finger stationary evidence for three consecutive
    # samples, with a bounded additional two seconds. Never relax the guard.
    stationary=0
    for attempt in range(21):
        stationary=stationary+1 if len(plant.contacts()) >= 2 else 0
        if stationary >= 3:break
        if attempt < 20:plant.step(.1)
    else:raise RuntimeError('Insufficient stable finger obstruction candidates')
    plant.set_contact_phase('lift')
    event('lift')
    plant.move(points[3], rotation=rotation)
    plant.step(.5)
    rises=[]
    previous_frame=None
    for sample in range(5):
        if sample:plant.step(.25)
        phase='lift-verification' if sample==0 else f'hold-verification-{sample}'
        lifted, vision = observe(phase, False, expected_center_xy=points[3][:2])
        frame_id=vision.get('frame_id')
        if frame_id is not None and frame_id==previous_frame:
            raise RuntimeError('Camera repeated a frame during lift hold verification')
        previous_frame=frame_id
        rise = vision['top_quantile_z_m']-baseline['top_quantile_z_m']
        if rise < .11 or np.linalg.norm(np.array(vision['center_xy_m'])-points[3][:2]) > .04:
            raise RuntimeError('Independent camera did not verify the lift and hold')
        rises.append(float(rise))
        event('lift_observation',sample=sample,visual_rise_m=float(rise),vision=vision)
    event('lift_verified', visual_rise_m=min(rises), hold_verified_s=1., vision=vision)
    return {'lift_verified': True, 'visual_rise_m': min(rises), 'hold_verified_s':1.}
