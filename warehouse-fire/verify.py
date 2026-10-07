"""Finite MuJoCo integration check, including a zone-boundary ignition."""
import json
import tempfile
from pathlib import Path
import numpy as np
from controller import FireController
from twin import ThermalTwin,zone_center

ROOT=Path(__file__).resolve().parent

def verify():
    with tempfile.TemporaryDirectory() as directory:
        twin=ThermalTwin(Path(directory)/'warehouse.xml')
        events=[]; active=set(); sources=[]; errors=[]; max_active=0
        covered=0; samples=0
        for x in np.arange(1,40,2):
            for y in np.arange(1,40,2):
                point=twin.surface(float(x),float(y)); samples+=1
                distances=np.nansum((twin.hits-point)**2,axis=-1)
                counts=np.sum((distances<1.0)&twin.valid,axis=(1,2))
                covered+=int(counts.max()>=2)
        assert covered==samples,(covered,samples)
        def emit(kind,**values): events.append(dict(kind=kind,**values))
        def valve(zone,on):
            if on: active.add(zone)
            else: active.discard(zone)
        controller=FireController(twin.hits,twin.cameras,emit,valve)
        for x,y in ((7.5,7.5),(30.4,23.5),(37.5,32.5),(30.,16.)):
            sources.append(dict(position=twin.surface(x,y),temperature=180.,burning=True))
        for tick in range(200):
            now=tick*.1
            for source in sources:
                water=any(np.linalg.norm(np.array(source['position'])[:2]-zone_center(z)[:2])<=3.6 for z in active)
                if source['burning']:
                    source['temperature']=max(22.,min(600.,source['temperature']+.1*(30-90*water)))
                    if water and source['temperature']<=35: source['burning']=False
                else: source['temperature']=max(22.,source['temperature']-3)
            frames=twin.thermal_frames(sources)
            controller.update(frames,now)
            twin.effects(sources,controller.incidents,active)
            max_active=max(max_active,len(active))
        for source in sources:
            closest=min(controller.incidents,key=lambda i:np.linalg.norm(np.array(i['position'])-source['position']))
            errors.append(float(np.linalg.norm(np.array(closest['position'])-source['position'])))
            assert errors[-1]<.8,errors
            assert np.linalg.norm(np.array(source['position'])[:2]-zone_center(closest['zone'])[:2])<=3.6
            assert not source['burning'],source
            assert source['temperature']==22,source
            assert closest['state']=='resolved',closest
        assert len(controller.incidents)==4,controller.incidents
        assert max_active==4,max_active
        assert not active,active
        assert twin.data.time>19.9,twin.data.time
        assert len([e for e in events if e['kind']=='sprinkler_on'])==4
        report=dict(passed=True,engine=mujoco_version(),cameras=len(twin.cameras),resolution=[32,24],warehouse_m=[40,40],fires=4,max_localization_error_m=round(max(errors),3),peak_active_zones=max_active,all_extinguished=True,all_valves_closed=True,physics_seconds=round(twin.data.time,2),coverage_samples=samples,covered_samples=covered,incidents=controller.incidents)
        (ROOT/'runtime').mkdir(exist_ok=True)
        (ROOT/'runtime/verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return report

def mujoco_version():
    import mujoco
    return 'MuJoCo '+mujoco.__version__

if __name__=='__main__': verify()
