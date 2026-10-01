"""Camera person detection and calibrated RGB/depth localization.

YOLOX-Tiny consumes rendered or physical RGB pixels. Person position comes
only from registered depth and the camera's capture-time optical transform.
Inference and MuJoCo rendering run on one worker, outside the safety tick.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from .types import Pose

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolox_tiny.onnx"


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray  # axial optical Z, metres; zero/nonfinite means invalid
    intrinsics: dict
    camera_to_world: np.ndarray  # optical right/down/forward -> output coordinates
    pose: Pose
    captured_at: float  # monotonic capture/receipt, not inference completion
    frame_id: str = "front_rgbd_optical"
    source_stamp: float | None = None  # ROS timestamp, when source is ROS

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must be a uint8 H×W×3 image")
        h,w = self.rgb.shape[:2]
        if h*w > 4_194_304 or w < 16 or h < 16 or self.depth.shape != (h,w):
            raise ValueError("RGB and aligned axial depth dimensions must agree")
        k = self.intrinsics
        if (k.get("width") != w or k.get("height") != h
                or not all(math.isfinite(float(k[v])) for v in ("fx","fy","cx","cy"))
                or k["fx"] <= 0 or k["fy"] <= 0
                or not 0 <= k["cx"] < w or not 0 <= k["cy"] < h):
            raise ValueError("Camera calibration does not match delivered pixels")
        matrix = np.asarray(self.camera_to_world)
        if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0,0,0,1], atol=1e-6)
                or not np.allclose(matrix[:3,:3].T@matrix[:3,:3], np.eye(3), atol=1e-4)
                or not math.isclose(float(np.linalg.det(matrix[:3,:3])), 1, abs_tol=1e-4)):
            raise ValueError("Camera optical transform must be a finite rigid transform")
        if not math.isfinite(self.captured_at) or not self.frame_id:
            raise ValueError("Camera frame requires capture time and optical frame")


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = .45) -> list[int]:
    if not len(boxes):
        return []
    areas = np.maximum(0, boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    order = scores.argsort()[::-1]
    keep = []
    while len(order) and len(keep) < 20:
        first = int(order[0]); keep.append(first)
        remaining = order[1:]
        intersection = np.maximum(0,np.minimum(boxes[first,2:],boxes[remaining,2:])-
                                  np.maximum(boxes[first,:2],boxes[remaining,:2])).prod(axis=1)
        overlap = intersection/np.maximum(areas[first]+areas[remaining]-intersection,1e-9)
        order = remaining[overlap <= threshold]
    return keep


class PersonDetector:
    """Official COCO YOLOX-Tiny ONNX model, CPU only, person class zero."""
    model_name = "YOLOX-Tiny COCO / CPU ONNX Runtime"

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = .35):
        import onnxruntime as ort
        if not model_path.is_file():
            raise FileNotFoundError(f"Person model is missing: {model_path}")
        if not .1 <= threshold <= .95:
            raise ValueError("Person confidence threshold must be within .1..95")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1,min(4,int(os.getenv("VISION_CPU_THREADS","2"))))
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[:2] != [1,3] or shape[2:] != [416,416]:
            raise ValueError("Bundled YOLOX-Tiny model must have 1×3×416×416 input")
        self.threshold = threshold
        self.sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        grids, strides = [], []
        for stride in (8,16,32):
            y,x = np.meshgrid(np.arange(416//stride),np.arange(416//stride),indexing="ij")
            grid = np.stack([x,y],axis=-1).reshape(-1,2)
            grids.append(grid); strides.append(np.full((len(grid),1),stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, rgb: np.ndarray) -> list[dict]:
        from PIL import Image
        height,width = rgb.shape[:2]
        scale = min(416/height,416/width)
        rw,rh = int(width*scale),int(height*scale)
        resized = np.asarray(Image.fromarray(rgb).resize((rw,rh),Image.Resampling.BILINEAR))
        # YOLOX release preprocessing is raw BGR, top-left letterbox, pad 114.
        padded = np.full((416,416,3),114,dtype=np.float32)
        padded[:rh,:rw] = resized[:,:,::-1]
        tensor = np.ascontiguousarray(padded.transpose(2,0,1)[None])
        output = self.session.run(None,{self.input.name:tensor})[0][0]
        if output.shape != (len(self.grid),85):
            raise ValueError("Unexpected YOLOX output shape")
        scores = output[:,4]*output[:,5]
        good = np.isfinite(output).all(axis=1)&(scores >= self.threshold)
        output,scores,grid,strides = output[good],scores[good],self.grid[good],self.strides[good]
        center = (output[:,:2]+grid)*strides
        size = np.exp(np.clip(output[:,2:4],-20,20))*strides
        boxes = np.column_stack((center-size/2,center+size/2))/scale
        boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,width-1)
        boxes[:,[1,3]] = np.clip(boxes[:,[1,3]],0,height-1)
        return [{"label":"person","confidence":round(float(scores[i]),4),
                 "bbox":[round(float(v),2) for v in boxes[i]]}
                for i in _nms(boxes,scores) if boxes[i,2]-boxes[i,0] >= 4 and boxes[i,3]-boxes[i,1] >= 8]


def ground_people(frame: CameraFrame, detections: list[dict]) -> list[dict]:
    """Localize detector boxes from registered depth; no depth means no position."""
    frame.validate()
    result = []
    height,width = frame.depth.shape
    for item in detections:
        x0,y0,x1,y1 = item["bbox"]
        if not all(math.isfinite(v) for v in (x0,y0,x1,y1)) or x1<=x0 or y1<=y0:
            continue
        # Central torso window reduces background/ground contamination.
        left,right = max(0,int(x0+.35*(x1-x0))),min(width,int(x0+.65*(x1-x0))+1)
        top,bottom = max(0,int(y0+.32*(y1-y0))),min(height,int(y0+.68*(y1-y0))+1)
        patch = frame.depth[top:bottom,left:right]
        valid = patch[np.isfinite(patch)&(patch>.15)&(patch<=8)]
        if valid.size < 12 or valid.size < patch.size*.25:
            continue
        depth = float(np.median(valid))
        # Reject strongly mixed depth instead of grounding a person on a wall.
        p10,p90 = np.percentile(valid,[10,90])
        if p90-p10 > max(.45,depth*.18):
            continue
        u,v = (x0+x1)/2,(top+bottom-1)/2
        k = frame.intrinsics
        optical = np.array([(u-k["cx"])*depth/k["fx"],
                            (v-k["cy"])*depth/k["fy"],depth,1])
        point = frame.camera_to_world@optical
        dx,dy = float(point[0])-frame.pose.x,float(point[1])-frame.pose.y
        distance = math.hypot(dx,dy)
        bearing = math.atan2(math.sin(math.atan2(dy,dx)-frame.pose.yaw),
                             math.cos(math.atan2(dy,dx)-frame.pose.yaw))
        result.append({**item,"x":float(point[0]),"y":float(point[1]),"z":float(point[2]),
                       "distance":distance,"bearing":bearing,"depth_m":depth,
                       "captured_at":frame.captured_at,"source":"rgb:yolox-tiny+aligned-depth"})
    return result


def annotate(frame: CameraFrame, detections: list[dict], people: list[dict]) -> bytes:
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(image)
    for i,item in enumerate(detections):
        bbox = item["bbox"]
        grounded = next((p for p in people if p["bbox"]==bbox),None)
        label = f"person {item['confidence']:.0%}"
        label += f"  {grounded['distance']:.2f}m" if grounded else "  depth unavailable"
        draw.rectangle(bbox,outline=(229,190,102),width=2)
        x,y = max(0,int(bbox[0])),max(0,int(bbox[1])-18)
        draw.rectangle((x,y,min(image.width,x+len(label)*6+8),y+16),fill=(27,34,26))
        draw.text((x+4,y+2),label,fill=(248,228,184))
    draw.rectangle((0,0,image.width,19),fill=(27,34,26))
    draw.text((7,4),"RGB camera | YOLOX-Tiny | measured aligned depth",fill=(248,228,184))
    output = io.BytesIO(); image.save(output,"JPEG",quality=82)
    return output.getvalue()


class MuJoCoCameraSource:
    """Worker-owned renderer from snapshot state, with actual camera calibration."""
    def __init__(self, simulation, width: int = 416, height: int = 312):
        self.simulation,self.width,self.height = simulation,width,height
        self._model = self._renderer = None

    def __call__(self) -> CameraFrame:
        import mujoco
        model,data,pose,captured_at = self.simulation.camera_snapshot()
        if self._model is not model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model,height=self.height,width=self.width)
            self._model = model
        camera_id = model.camera("front_rgbd").id
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0  # physical human capsules are not camera pixels
        options.geomgroup[4] = 1  # textured human meshes are the camera's subjects
        options.geomgroup[5] = 1  # the real floor belongs in both RGB and depth
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data,camera=camera_id,scene_option=options)
        # Software rasterization does not need a second shadow-map render pass.
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data,camera=camera_id,scene_option=options)
        depth = self._renderer.render().astype(np.float32).copy()
        self._renderer.disable_depth_rendering()
        fovy = math.radians(float(model.cam_fovy[camera_id]))
        focal = self.height/(2*math.tan(fovy/2))
        k = {"fx":focal,"fy":focal,"cx":(self.width-1)/2,"cy":(self.height-1)/2,
             "width":self.width,"height":self.height}
        camera_rotation = data.cam_xmat[camera_id].reshape(3,3)
        optical_to_world = np.eye(4)
        optical_to_world[:3,:3] = camera_rotation@np.diag([1,-1,-1])
        optical_to_world[:3,3] = data.cam_xpos[camera_id]
        return CameraFrame(rgb,depth,k,optical_to_world,pose,captured_at)

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None


class CameraPerception:
    """Bounded newest-frame worker. Snapshot never runs detection or rendering."""
    def __init__(self, source: Callable[[],CameraFrame|None], interval: float = .2,
                 detector_factory: Callable[[],PersonDetector] = PersonDetector):
        self.source,self.interval,self.detector_factory = source,max(.05,interval),detector_factory
        self._lock = threading.Lock(); self._stop = threading.Event(); self._thread = None
        self._snapshot = {"ready":False,"state":"loading","error":"","model":"YOLOX-Tiny COCO",
            "frame_id":"","captured_at":None,"inference_ms":None,"people":[],"detections":[],"jpeg":None}
        self._last_capture = -math.inf; self._tracks = {}; self._next_id = 1

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run,name="rgb-person-perception",daemon=True)
            self._thread.start()
        return self

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._snapshot,"people":[dict(p) for p in self._snapshot["people"]],
                    "detections":[dict(p) for p in self._snapshot["detections"]]}

    def _store(self, **values):
        with self._lock:
            self._snapshot = {**self._snapshot,**values}

    def _track(self, people, capture):
        old = {i:p for i,p in self._tracks.items() if capture-p["captured_at"] < 2}
        used = set()
        for person in people:
            candidates = [(math.hypot(person["x"]-p["x"],person["y"]-p["y"]),i)
                          for i,p in old.items() if i not in used]
            best = min(candidates,default=(math.inf,None))
            if best[0]<.7:
                identity=best[1]
            else:
                identity=f"camera-{self._next_id}"; self._next_id+=1
            person["id"]=identity; used.add(identity)
        # Keep identity through a brief missed box, without reporting that
        # absent track as a current detection or inventing its position.
        self._tracks={**old, **{p["id"]:p for p in people}}

    def _run(self):
        try:
            detector = self.detector_factory()
            self._store(model=detector.model_name,model_sha256=detector.sha256)
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    frame = self.source()
                    if frame is None:
                        self._store(ready=False,state="waiting",error="Waiting for synchronized camera pixels and calibration",people=[],detections=[],jpeg=None)
                    elif frame.captured_at > self._last_capture:
                        frame.validate()
                        rendered = time.monotonic()
                        detections = detector.detect(frame.rgb)
                        detected = time.monotonic()
                        people = ground_people(frame,detections)
                        self._track(people,frame.captured_at)
                        jpeg = annotate(frame,detections,people)
                        self._last_capture = frame.captured_at
                        self._store(ready=True,state="observed",error="",frame_id=frame.frame_id,
                            captured_at=frame.captured_at,source_stamp=frame.source_stamp,
                            inference_ms=round((time.monotonic()-started)*1000,1),
                            render_ms=round((rendered-started)*1000,1),
                            detect_ms=round((detected-rendered)*1000,1),
                            people=people,detections=detections,jpeg=jpeg)
                except Exception as error:
                    self._store(ready=False,state="error",error=f"{type(error).__name__}: {error}",
                                people=[],detections=[],jpeg=None)
                self._stop.wait(max(0,self.interval-(time.monotonic()-started)))
        except Exception as error:
            self._store(ready=False,state="error",error=f"{type(error).__name__}: {error}",people=[],detections=[],jpeg=None)
        finally:
            if hasattr(self.source,"close"):
                self.source.close()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
