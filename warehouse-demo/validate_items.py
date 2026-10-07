"""Held-out camera validation. Known object positions are test oracles only."""
import json
import numpy as np
import mujoco,cv2
from ultralytics import YOLO
from demo import ROOT,W,H,make_scene,localize,approach
from items import ITEMS,NAMES

def validate():
    detector=YOLO(str(ROOT/'runtime/items-yolo.pt'))
    layouts=[{i.name:np.array(i.position) for i in ITEMS}]
    rng=np.random.default_rng(56382)
    slots=rng.permutation(10)
    layouts.append({i.name:np.array(ITEMS[int(slots[j])].position)+rng.uniform(-.15,.15,2) for j,i in enumerate(ITEMS)})
    reports=[]
    for layout in layouts:
        model=make_scene(objects=True,positions=layout);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        found={}
        with mujoco.Renderer(model,height=H,width=W) as renderer:
            for camera in ('cam1','cam2','cam3','cam4'):
                renderer.update_scene(data,camera=camera)
                image=cv2.cvtColor(renderer.render().copy(),cv2.COLOR_RGB2BGR)
                image=cv2.imdecode(cv2.imencode('.jpg',image)[1],cv2.IMREAD_COLOR)
                prediction=detector.predict(image,conf=.4,imgsz=960,device='cpu',verbose=False)[0]
                for box in prediction.boxes:
                    name=NAMES[int(box.cls[0])];hit=localize(model,data,camera,box.xyxy[0].cpu().numpy())
                    if hit is None:continue
                    error=float(np.linalg.norm(hit[:2]-layout[name]))
                    try:approach((0,0),hit[:2])
                    except RuntimeError:continue
                    result=dict(camera=camera,confidence=float(box.conf[0]),ray_error_m=error)
                    if name not in found or error<found[name]['ray_error_m']:found[name]=result
                cv2.imwrite(str(ROOT/f'runtime/items-validation-{len(reports)}-{camera}.jpg'),prediction.plot())
        reports.append(found)
    report=dict(passed=all(len(r)==10 and all(v['ray_error_m']<.5 for v in r.values()) for r in reports),
                layouts=reports,physical_world_validated=False)
    (ROOT/'runtime/items-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert report['passed'],'Some items lack a confident, correctly localized detection'

if __name__=='__main__':validate()
