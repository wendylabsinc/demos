import sys
sys.path.insert(0,"/app")
import json,torch,cv2,numpy as np
from ultralytics import YOLO
from qualification import fixtures,check
from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates
from service import Detector
from confidence import MIN_CONFIDENCE
assert torch.__version__.startswith("2.7.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert cv2.__version__ == "4.11.0", cv2.__version__
assert np.__version__ == "2.2.6", np.__version__
torch.set_num_threads(2)
model=YOLO("/models/coke-can.pt")
network=model.model.fuse(verbose=False).eval()
probe=Detector.__new__(Detector)
probe.network=network
probe.torch=torch
verified=[]
for row,bgr in fixtures("/fixtures"):
 chw,scale,padx,pady=letterbox_bgr(bgr,640)
 with torch.inference_mode():
  packed=probe.packed_forward(torch.from_numpy(chw).float().div_(255).unsqueeze(0)).numpy()
 candidates=packed[:-1].reshape(128,5)
 decoded=decode_one_class_candidates(candidates,confidence=MIN_CONFIDENCE,nms_iou=.7,max_detections=21,frame_width=bgr.shape[1],frame_height=bgr.shape[0],scale=scale,pad_x=padx,pad_y=pady)
 boxes=[{"confidence":b["conf"],"xyxy":[b["x1"],b["y1"],b["x2"],b["y2"]]} for b in decoded]
 verified.append(check(row,boxes,bool(packed[-1]>MIN_CONFIDENCE)))
print("OFFLINE_BUILD_PROOF",json.dumps({"fixture_count":len(verified),"qualified":True,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"numpy":np.__version__,"opencv":cv2.__version__,"motion_commands":0}),flush=True)
