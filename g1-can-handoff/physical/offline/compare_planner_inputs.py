"""Compare frozen planner settings on identical saved-scene inputs, no commands."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def evaluate(index,policy):
    import numpy as np
    from offline.planning_failures_replay import load_case
    from offline.planner_replay import clearance_auditor
    from planner_tuning import configure
    from pickup import prepare
    plant,observation,events,provenance=load_case(index)
    inputs=dict(measured=plant.read(),qpos=plant.ik.data.qpos.tolist(),qvel=plant.ik.data.qvel.tolist(),
        model_body_pos=plant.ik.model.body_pos.tolist(),model_geom_pos=plant.ik.model.geom_pos.tolist(),
        center_m=observation.center.tolist(),table_height_m=observation.table_height,
        config={k:v for k,v in plant.config.items() if k not in ('ik_reference_weight_m_per_rad','ik_position_tolerance_m','ik_variable_scale','grasp_orientation_step_rad','grasp_candidate_budget','planning_position_cov_m2')})
    digest=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    profile=(json.loads((ROOT/'g1_can_handoff/noise_aware_planner.json').read_text()) if policy=='tuned' else
        dict(ik_reference_weight_m_per_rad=0.,ik_position_tolerance_m=.012,ik_variable_scale='unit',grasp_orientation_step_rad=0.,grasp_candidate_budget=1))
    plant.config.update(profile);configure(plant,plant.config)
    clearance=clearance_auditor(plant);before=plant.ik.data.qpos.copy();started=time.monotonic()
    try:
        _,_,points,rotation=prepare(plant,lambda *args:(observation,dict(source='identical_saved_scene')),plant.event)
        np.testing.assert_array_equal(plant.ik.data.qpos,before)
        result=dict(passed=True,points=[p.tolist() for p in points],rotation=rotation.tolist())
    except (ValueError,RuntimeError,AssertionError) as exc:result=dict(passed=False,error=str(exc))
    result.update(policy=policy,input_sha256=digest,provenance=provenance,clearance=clearance,events=events,
        elapsed_s=time.monotonic()-started,hardware_used=False,physical_pickup_verified=False,
        adapters_absent=plant.adapter is None and plant.sender is None)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=int,choices=range(6),required=True)
    parser.add_argument('--policy',choices=('legacy','tuned'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=evaluate(args.case,args.policy)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('policy','passed','elapsed_s','input_sha256','adapters_absent')}),flush=True)
