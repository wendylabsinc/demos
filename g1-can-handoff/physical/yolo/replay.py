"""Reproduce trained-model recognition on a saved frame; never contacts a robot.

CPU is intentional for offline replay and is not a physical-runtime fallback.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates


def main():
    root=Path(__file__).parent
    manifest=json.loads((root/'model/SOURCE.json').read_text())
    weights=root/'model/coke-can.pt'
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest!=manifest['sha256']:raise RuntimeError('Checkpoint checksum mismatch')
    annotation=json.loads((root/'fixtures/g1-coke-raw.json').read_text())
    image=np.asarray(Image.open(root/'fixtures/g1-coke-raw.jpg').convert('RGB'))[:,:,::-1].copy()
    model=YOLO(weights)
    if model.names!={0:'Coca-Cola'}:raise RuntimeError('Expected Coca-Cola checkpoint')
    result=model.predict(image,device='cpu',imgsz=640,rect=False,conf=.3,verbose=False)[0]
    expected=np.asarray(annotation['bbox_xyxy'])
    candidates=[]
    for box in result.boxes:
        if float(box.conf[0]) <= .30:
            continue
        observed=box.xyxy[0].numpy()
        intersection=np.maximum(0,np.minimum(observed[2:],expected[2:])-np.maximum(observed[:2],expected[:2])).prod()
        union=np.prod(observed[2:]-observed[:2])+np.prod(expected[2:]-expected[:2])-intersection
        candidates.append(dict(xyxy=observed.tolist(),confidence=float(box.conf[0]),iou=float(intersection/union)))
    if len(candidates)!=1 or candidates[0]['iou']<.5:
        raise RuntimeError('Saved real Coke fixture failed semantic recognition: '+str(candidates))
    chw,scale,pad_x,pad_y=letterbox_bgr(image,640)
    with torch.inference_mode():
        raw=model.model(torch.from_numpy(chw).float().div_(255).unsqueeze(0))
    if isinstance(raw,(tuple,list)):raw=raw[0]
    if tuple(raw.shape)!=(1,5,8400):raise RuntimeError('Unexpected raw one-class output')
    _,indices=raw[0,4].topk(129,largest=True,sorted=True)
    decoded=decode_one_class_candidates(raw[0,:,indices[:128]].T.numpy(),confidence=.3,
        nms_iou=.7,max_detections=21,frame_width=image.shape[1],frame_height=image.shape[0],
        scale=scale,pad_x=pad_x,pad_y=pad_y)
    if len(decoded)!=1 or decoded[0]['conf']<=.30:
        raise RuntimeError('Optimized one-class decoding failed fixture')
    optimized=np.asarray([decoded[0][k] for k in ('x1','y1','x2','y2')])
    if np.max(np.abs(optimized-np.asarray(candidates[0]['xyxy'])))>1 or abs(decoded[0]['conf']-candidates[0]['confidence'])>.01:
        raise RuntimeError('Optimized decoding does not preserve Ultralytics box/confidence')
    print(json.dumps(dict(checkpoint_sha256=digest,offline_fixture_passed=True,
                         backend='cpu_offline_replay',detections=candidates,optimized_decoder_parity_passed=True,
                         physical_pickup_verified=False),indent=2))


if __name__=='__main__':main()
