"""Train YOLO on rendered props. Simulator IDs generate training labels only.
Live search uses the resulting image detector, never these IDs or positions.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import mujoco
from ultralytics import YOLO
from PIL import Image
import yaml
from items import ITEMS, NAMES
from demo import ROOT, W, H, make_scene

def dataset(count=40):
    path=ROOT/'runtime/item-dataset'
    if path.exists(): shutil.rmtree(path)
    for split in ('train','val'):
        for kind in ('images','labels'): (path/kind/split).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(91421)
    for scene in range(count):
        slots=rng.permutation(len(ITEMS))
        positions={item.name:np.array(ITEMS[int(slots[i])].position)+rng.uniform(-.23,.23,2) for i,item in enumerate(ITEMS)}
        model=make_scene(objects=True,positions=positions)
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        split='val' if scene%5==0 else 'train'
        gid_class={i:NAMES.index(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i]))[5:])
                   for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or '').startswith('item_')}
        with mujoco.Renderer(model,height=H,width=W) as renderer:
            for cam in range(4):
                cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,f'cam{cam+1}')
                model.cam_fovy[cid]=rng.uniform(62,68)
                renderer.update_scene(data,camera=f'cam{cam+1}')
                rgb=renderer.render().copy()
                renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
                labels=[]
                for cls in range(10):
                    ids=[gid for gid,c in gid_class.items() if c==cls]
                    mask=np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                    yy,xx=np.where(mask)
                    if len(xx)<35: continue
                    xmin,xmax=max(0,xx.min()-2),min(W,xx.max()+3);ymin,ymax=max(0,yy.min()-2),min(H,yy.max()+3)
                    labels.append(f'{cls} {(xmin+xmax)/2/W:.6f} {(ymin+ymax)/2/H:.6f} {(xmax-xmin)/W:.6f} {(ymax-ymin)/H:.6f}')
                stem=f'{scene:03d}-{cam}'
                Image.fromarray(rgb).save(path/'images'/split/(stem+'.jpg'))
                (path/'labels'/split/(stem+'.txt')).write_text('\n'.join(labels))
        if scene%10==0: print('Rendered scenes',scene,flush=True)
    config=path/'data.yaml';config.write_text(yaml.safe_dump(dict(path=str(path),train='images/train',val='images/val',names=dict(enumerate(NAMES)))))
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=35);parser.add_argument('--scenes',type=int,default=40)
    args=parser.parse_args()
    config=dataset(args.scenes)
    import torch
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=YOLO(str(ROOT/'yolo11n.pt'))
    model.train(data=str(config),epochs=args.epochs,imgsz=640,batch=8,device=device,workers=0,
        project=str(ROOT/'runtime/training'),name='items',exist_ok=True,seed=2718,
        fliplr=.5,flipud=0,mosaic=.5,close_mosaic=5,patience=12,plots=False)
    shutil.copyfile(model.trainer.best,ROOT/'runtime/items-yolo.pt')
    (ROOT/'runtime/detector-source.json').write_text(json.dumps(dict(
        labels=NAMES,kind='YOLO11n fine-tuned on simulated camera images',
        dataset=str(config),seed=91421,epochs=args.epochs,physical_world_validated=False),indent=2))
    print('Detector ready:',ROOT/'runtime/items-yolo.pt')
