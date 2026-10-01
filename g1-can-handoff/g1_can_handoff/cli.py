import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .sim import scene
from .perception import SimCamera
from .task import execute

def trial(height,x,y,mass=.37,friction=1.,output=None,record=False,depth_noise=0.,view="observe"):
    plant=scene(height,x,y,mass,friction)
    plant.step(.3)
    camera=SimCamera(plant,depth_noise)
    result={"table_height":height,"x":x,"y":y,"mass":mass,"friction":friction,"depth_noise_m":depth_noise}
    frames=[];last_frame=[-1.]
    try:
        observation,rgb=camera.observe()
        result["position_error_m"]=float(np.linalg.norm(observation.center-plant.can_position()))
        result["table_error_m"]=abs(observation.table_height-height)
        if output:
            Path(output).mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(Path(output)/"camera.png")
            if view!="observe":
                camera.renderer.update_scene(plant.data,camera=view)
                Image.fromarray(camera.renderer.render().copy()).save(Path(output)/"view.png")
        def frame(p):
            if p.data.time-last_frame[0]<.10:return
            last_frame[0]=p.data.time
            camera.renderer.update_scene(p.data,camera=view)
            frames.append(Image.fromarray(camera.renderer.render().copy()))
        if record:plant.frame_callback=frame
        phases=[]
        outcome=execute(plant,observation,phases.append,lambda:camera.observe()[0])
        result.update(outcome,success=True,phases=phases)
        if record:result["recording_camera"]=view
    except ValueError as exc:
        result.update(success=False,error=str(exc),can_height=float(plant.can_position()[2]))
    finally:
        if frames and output:
            frames[0].save(Path(output)/"pickup.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
        camera.close()
    return result

def main():
    parser=argparse.ArgumentParser(description="G1 Dex3 RGB-D can pickup simulation")
    parser.add_argument("command",choices=["demo","evaluate"],nargs="?",default="demo")
    parser.add_argument("--height",type=float,default=.75)
    parser.add_argument("--x",type=float,default=.35)
    parser.add_argument("--y",type=float,default=.18)
    parser.add_argument("--output",default="results")
    parser.add_argument("--random-cases",type=int,default=0)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--record",action="store_true")
    parser.add_argument("--view",choices=["observe","head"],default="observe",help="Recording viewpoint; perception still uses the observe camera")
    parser.add_argument("--boundaries",action="store_true",help="Also test lower/farther targets outside the validated rectangular pickup area")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.command=="demo":
        result=trial(args.height,args.x,args.y,output=output,record=args.record,view=args.view)
        (output/"demo.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0 if result["success"] else 1)
    cases=list(itertools.product([.75,.85,.95],[.28,.315,.35],[.10,.19,.28]))
    if args.boundaries:cases.extend(itertools.product([.65,.75],[.35,.44],[.10,.19,.28]))
    rng=np.random.default_rng(args.seed)
    cases.extend((float(rng.uniform(.75,.95)),float(rng.uniform(.28,.35)),float(rng.uniform(.10,.28))) for _ in range(args.random_cases))
    results=[]
    for i,(h,x,y) in enumerate(cases):
        varied=i>=len(cases)-args.random_cases and args.random_cases>0
        result=trial(h,x,y,mass=float(rng.uniform(.33,.40)) if varied else .37,friction=float(rng.uniform(.7,1.2)) if varied else 1.,depth_noise=.001 if varied else 0.)
        results.append(result)
        print(f"{i+1}/{len(cases)} height={h:.3f} x={x:.3f} y={y:.3f}: "+("PASS" if result["success"] else result["error"]),flush=True)
        (output/"evaluation.json").write_text(json.dumps({"seed":args.seed,"passed":sum(r["success"] for r in results),"total":len(cases),"completed":len(results),"cases":results},indent=2))
    raise SystemExit(0 if all(r["success"] for r in results) else 1)

if __name__=="__main__":main()
