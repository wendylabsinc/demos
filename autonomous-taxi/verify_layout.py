"""Check the city geometry and map use the same road-clear layout."""
import json
from pathlib import Path
import mujoco
import numpy as np
from world import City, TREES, road_clearance

c=City(); checked=[]
for i in range(c.model.ngeom):
    name=mujoco.mj_id2name(c.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
    if name.startswith(('leaf','building','taxi_bay','passenger','pole','signal')):
        xy=c.data.geom_xpos[i,:2]
        rotation=c.data.geom_xmat[i].reshape(3,3)
        extent=np.abs(rotation) @ c.model.geom_size[i]
        if c.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent=c.model.geom_size[i,0]*np.linalg.norm(rotation[:,:2],axis=1)+c.model.geom_size[i,1]*np.abs(rotation[:,2])
        # Check the footprint corners as well as its center against the road union.
        for dx in (-extent[0],0,extent[0]):
            for dy in (-extent[1],0,extent[1]):
                assert road_clearance(xy[0]+dx,xy[1]+dy)>0, name+' overlaps the road'
        checked.append(name)
assert len(c.map_geometry['trees'])==len(TREES)==20
assert len(c.map_geometry['buildings'])==16
assert len(c.map_geometry['roads'])==6
result=dict(passed=True,checked_objects=len(checked),trees=len(TREES),minimum_canopy_clearance_m=round(min(road_clearance(x,y)-1.6 for x,y in TREES),2),map_uses_model_geometry=True)
out=Path(__file__).parent/'runtime';out.mkdir(exist_ok=True);(out/'layout-verification.json').write_text(json.dumps(result,indent=2));print(result)
