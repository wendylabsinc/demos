"""Predeclared complete-route checks in two changed pickup scenes."""
import argparse,json,contextlib,time
import numpy as np
from . import bar_delivery_physics as delivery
from . import bar_film as bar

CASES={
    'full75':dict(height=.75,x=.28,y=.10,delivery_height=.75),
    'full95':dict(height=.95,x=.35,y=.28,delivery_height=.75),
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=CASES);args=parser.parse_args()
    bar.textures();delivery.OUTPUT=bar.PROJECT/'results'/'generalization-physics'/args.case
    delivery.OUTPUT.mkdir(parents=True,exist_ok=True)
    scenario=CASES[args.case];state={};start=time.monotonic()
    def retain(p,frames,events):state.update(plant=p,frames=frames,events=events)
    with (delivery.OUTPUT/'log.txt').open('w') as log,contextlib.redirect_stdout(log):
        try:
            p,frames=delivery.simulate(**scenario,on_start=retain)
            result=json.loads((delivery.OUTPUT/'simulation.json').read_text())
        except Exception as error:
            p=state.get('plant');frames=state.get('frames',[])
            result=dict(success=False,failure=str(error),failure_type=type(error).__name__,events=state.get('events',[]))
            if p is not None:
                result.update(min_pelvis_height=p.min_height,min_upright_cosine=p.min_upright,max_robot_furniture_contact_force_N=p.max_furniture_force,
                    simulation_duration_s=float(p.data.time),physics_only=p.model.neq==0 and p.model.nmocap==0,
                    support_constraints=int(p.model.neq),mocap_bodies=int(p.model.nmocap))
            if frames:np.savez_compressed(delivery.OUTPUT/'states.npz',**{k:np.stack([f[k] for f in frames]) for k in frames[0]})
    result.update(id=args.case,kind='full_route',scenario=scenario,wall_time_s=time.monotonic()-start)
    (delivery.OUTPUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
