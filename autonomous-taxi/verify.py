"""Run actual MuJoCo physics, mission, sensor, and braking regression checks."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from world import City, OUTER, DT, CAMERAS


def run():
    c=City()
    initial=c.data.qpos.copy()
    with patch.object(mujoco,'mj_step'):
        for _ in range(10): c.step()
    assert np.array_equal(initial,c.data.qpos), 'Controller must not assign vehicle poses'
    assert c.model.nmocap==0
    assert c.model.ncam==len(CAMERAS)==6
    assert np.count_nonzero(c.scan<60)>20, 'LiDAR must hit actual geometry'
    red_crossings=[]; crossings=0
    lines=[]
    for p,vertical in [([0,-38.4],False),([38.4,0],True),([0,38.4],False),([-38.4,0],True)]:
        lines.append(((OUTER.nearest(np.array(p))[0]-9)%OUTER.length,vertical))
    for _ in range(int(300/DT)):
        old=c.cars[0]['s']; c.step(); new=c.cars[0]['s']; traveled=(new-old)%OUTER.length
        if traveled<1:
            for line,vertical in lines:
                if 0<(line-old)%OUTER.length<=traveled:
                    crossings+=1
                    if not c.light_green(vertical): red_crossings.append(c.data.time)
        assert np.isfinite(c.data.qpos).all() and np.isfinite(c.data.qvel).all()
    assert c.completed>=2, f'Expected completed taxi trips, got {c.completed}'
    assert c.sign_stops>=2, 'Expected multiple full STOP sign stops'
    assert c.red_wait>5, 'Expected stationary red-light waits'
    assert not red_crossings, f'Crossed red stop lines at {red_crossings}'
    assert c.collisions==0, f'Taxi collisions: {c.collisions}'
    assert c.max_error<1.1, f'Lane tracking error {c.max_error}'
    mission=dict(simulation_seconds=round(c.data.time,2),trips=c.completed,stop_sign_stops=c.sign_stops,red_wait_seconds=round(c.red_wait,2),signal_crossings=crossings,red_violations=len(red_crossings),collisions=c.collisions,max_lane_error_m=round(c.max_error,3),distance_m=round(c.distance),events=c.events)
    # Place traffic at a known INITIAL position, then let MuJoCo integrate it.
    blocked=City(); blocked.mission=1
    car=blocked.cars[1]; p,yaw=OUTER.at(25.)
    blocked.data.qpos[car['q']:car['q']+3]=[*p,yaw]; car['s']=25.; car['progress']=25.; car['parked']=True
    mujoco.mj_forward(blocked.model,blocked.data); blocked.update_sensors()
    for _ in range(2200): blocked.step()
    q=blocked.cars[0]['q']; gap=np.linalg.norm(blocked.data.qpos[q:q+2]-blocked.data.qpos[car['q']:car['q']+2])-4.1
    assert abs(blocked.cars[0]['speed'])<.15, 'LiDAR obstacle must stop taxi'
    assert blocked.lidar_wait>4 and gap>1.5 and blocked.collisions==0, f'Unsafe obstacle gap {gap}'
    parked_state=blocked.state()
    car['parked']=False
    for _ in range(2400): blocked.step()
    assert blocked.distance>parked_state['distance']+10, 'Taxi must resume when traffic moves'
    # Missing range updates must brake, independently of route information.
    with patch.object(blocked,'update_sensors'):
        for _ in range(400): blocked.step()
    assert blocked.cars[0]['reason']=='Sensor fault' and abs(blocked.cars[0]['speed'])<.05
    blocked.update_sensors(); blocked.emergency=True
    for _ in range(300): blocked.step()
    assert abs(blocked.cars[0]['speed'])<.05
    result=dict(passed=True,engine=mujoco.__version__,physics='MuJoCo planar dynamic bicycle tire forces',mission=mission,lidar=dict(rays=len(c.scan),max_range_m=60,obstacle_gap_m=round(gap,2),stale_sensor_braking=True),camera_names=CAMERAS,no_pose_teleportation=True,emergency_braking=True)
    out=Path(__file__).parent/'runtime'; out.mkdir(exist_ok=True); (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': run()
