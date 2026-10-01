"""ROS 2 RGB/depth bridge for the same local camera person detector.

Run with ``python -m sassy_rosmaster.perception_ros --config perception.json``.
This process subscribes to explicitly configured, rectified RGB and registered
axial depth, uses calibrated CameraInfo and a measured fixed optical-to-base
transform, and publishes measured people to the hardware adapter. It never
publishes drive commands. Missing/stale/uncalibrated images produce no heartbeat.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from .perception import CameraFrame, CameraPerception
from .types import Pose


def validate_config(config: dict) -> dict:
    if not isinstance(config,dict):
        raise ValueError("Perception configuration must be an object")
    result = dict(config)
    if config.get("calibration_verified") is not True:
        raise ValueError("Measure camera calibration and set calibration_verified before running the bridge")
    for key in ("rgb_rectified","depth_registered_to_rgb","fixed_camera_mount"):
        if config.get(key) is not True:
            raise ValueError(f"Verify {key}; this bridge does not rectify/register images or guess a moving mount")
    for key in ("rgb_topic","depth_topic","camera_info_topic","people_topic"):
        value = config.get(key)
        if not isinstance(value,str) or not value.startswith("/") or any(c.isspace() for c in value):
            raise ValueError(f"Set an explicit absolute {key}")
    for key in ("optical_frame","base_frame"):
        if not isinstance(config.get(key),str) or not config[key].strip():
            raise ValueError(f"Set the measured {key}")
    if config["optical_frame"] == config["base_frame"]:
        raise ValueError("Camera optical and robot base frames must differ")
    transform = np.asarray(config.get("camera_to_base"),dtype=float)
    if (transform.shape != (4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1],atol=1e-6)
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4)
            or not math.isclose(float(np.linalg.det(transform[:3,:3])),1,abs_tol=1e-4)):
        raise ValueError("Set a measured rigid 4×4 camera_to_base optical transform in metres")
    result["camera_to_base"] = transform
    for key,low,high in (("sync_slop",.001,.05),("max_age",.05,.3),("inference_interval",.05,.2)):
        value = config.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f"Set {key} within {low}..{high} seconds")
    if not isinstance(config.get("camera_info_static"),bool):
        raise ValueError("Explicitly set camera_info_static for the camera driver's calibration timestamps")
    if not isinstance(config.get("use_sim_time"),bool):
        raise ValueError("Explicitly set use_sim_time to match the source image clock")
    return result


def stamp_seconds(header, *, allow_zero: bool = False) -> float:
    stamp = header.stamp
    if (not isinstance(stamp.sec,int) or not isinstance(stamp.nanosec,int)
            or stamp.sec<0 or not 0<=stamp.nanosec<1_000_000_000):
        raise ValueError("Invalid ROS capture timestamp")
    seconds = stamp.sec+stamp.nanosec*1e-9
    if seconds<=0 and not allow_zero:
        raise ValueError("ROS image capture timestamp is missing")
    return seconds


def decode_image(message) -> np.ndarray:
    """Honor ROS Image encoding, padded row strides and endianness."""
    formats = {"rgb8":("u1",3),"bgr8":("u1",3),"32FC1":("f4",1),"16UC1":("u2",1)}
    if message.encoding not in formats:
        raise ValueError("Use rgb8/bgr8 RGB and 16UC1-mm or 32FC1-m aligned depth")
    code,channels = formats[message.encoding]
    dtype = np.dtype((">" if message.is_bigendian else "<")+code)
    w,h,step = message.width,message.height,message.step
    if (not all(isinstance(v,int) for v in (w,h,step)) or not 16<=w<=4096 or not 16<=h<=4096
            or w*h>4_194_304 or step<w*channels*dtype.itemsize
            or step*h>64*1024*1024 or len(message.data)!=step*h):
        raise ValueError("Malformed or oversized camera image dimensions/stride")
    pixels = np.ndarray((h,w,channels),dtype=dtype,buffer=bytes(message.data),
        strides=(step,channels*dtype.itemsize,dtype.itemsize)).copy()
    if channels==1:
        pixels=pixels[:,:,0].astype(np.float32)
        if message.encoding=="16UC1":
            pixels*=.001
    elif message.encoding=="bgr8":
        pixels=pixels[:,:,::-1].copy()
    return pixels


def camera_intrinsics(info) -> dict:
    """Use rectified P; reject unsupported rectification, crops and stereo offsets."""
    p,r = list(info.p),list(info.r)
    expected=[1,0,0,0,1,0,0,0,1]
    if (len(p)!=12 or len(r)!=9 or not all(math.isfinite(v) for v in (*p,*r))
            or p[0]<=0 or p[5]<=0
            or any(abs(a-b)>1e-5 for a,b in zip(r,expected))
            or any(abs(p[i])>1e-5 for i in (1,3,4,7,8,9,11)) or abs(p[10]-1)>1e-5):
        raise ValueError("CameraInfo needs a calibrated rectified optical P, identity R and zero stereo offset")
    roi=info.roi
    if (info.binning_x not in (0,1) or info.binning_y not in (0,1)
            or roi.x_offset!=0 or roi.y_offset!=0
            or roi.width not in (0,info.width) or roi.height not in (0,info.height)):
        raise ValueError("Express cropped/binned calibration for the delivered image pixels first")
    w,h=info.width,info.height
    if (not isinstance(w,int) or not isinstance(h,int) or not 16<=w<=4096 or not 16<=h<=4096
            or w*h>4_194_304 or not 0<=p[2]<w or not 0<=p[6]<h):
        raise ValueError("CameraInfo dimensions and principal point are invalid")
    return {"width":w,"height":h,"fx":p[0],"fy":p[5],"cx":p[2],"cy":p[6]}


class RosCameraSource:
    """Bounded synchronizer. ROS callbacks cache pixels; the worker pulls frames."""
    def __init__(self, config: dict, ros_now, *, monotonic=time.monotonic):
        self.config=validate_config(config); self.ros_now=ros_now; self.monotonic=monotonic
        self._lock=threading.Lock()
        self._cache={kind:deque(maxlen=4) for kind in ("rgb","depth","camera_info")}
        self._last_stamp=-math.inf
        self.error="Waiting for synchronized RGB, registered depth and CameraInfo"

    def receive(self, kind: str, message) -> None:
        try:
            if message.header.frame_id!=self.config["optical_frame"]:
                raise ValueError("Camera input frame differs from calibrated optical_frame")
            static=kind=="camera_info" and self.config["camera_info_static"]
            stamp=stamp_seconds(message.header,allow_zero=static)
            age=self.ros_now()-stamp
            if not static and not -.05<=age<=self.config["max_age"]:
                raise ValueError("Camera capture is stale or the ROS clocks are unsynchronized")
            if kind=="camera_info":
                value=camera_intrinsics(message)
            else:
                expected=("rgb8","bgr8") if kind=="rgb" else ("16UC1","32FC1")
                if message.encoding not in expected:
                    raise ValueError(f"Unsupported {kind} encoding")
                value=decode_image(message)
            with self._lock:
                self._cache[kind].append((stamp,value))
                self.error=""
        except (ValueError,AttributeError,TypeError,OverflowError) as error:
            with self._lock:
                self._cache[kind].clear()
                self.error=str(error)

    def __call__(self) -> CameraFrame | None:
        with self._lock:
            if any(not items for items in self._cache.values()):
                return None
            for stamp,rgb in reversed(self._cache["rgb"]):
                if stamp<=self._last_stamp:
                    continue
                now=self.ros_now()
                age=now-stamp
                if not -.05<=age<=self.config["max_age"]:
                    self.error="RGB capture expired before inference"
                    return None
                depth_stamp,depth=min(self._cache["depth"],key=lambda item:abs(item[0]-stamp))
                info_stamp,k=min(self._cache["camera_info"],key=lambda item:abs(item[0]-stamp))
                if abs(depth_stamp-stamp)>self.config["sync_slop"]:
                    self.error="Waiting for depth synchronized to RGB capture"
                    continue
                depth_age=now-depth_stamp
                if not -.05<=depth_age<=self.config["max_age"]:
                    self.error="Depth capture expired before inference or clocks are unsynchronized"
                    continue
                if not self.config["camera_info_static"] and abs(info_stamp-stamp)>self.config["sync_slop"]:
                    self.error="Waiting for capture-time CameraInfo"
                    continue
                if depth.shape!=rgb.shape[:2] or (k["height"],k["width"])!=rgb.shape[:2]:
                    self.error="RGB, aligned depth and CameraInfo dimensions differ"
                    return None
                # Preserve RGB's outgoing ROS stamp while freshness covers both
                # measured pixel modalities, including an older aligned depth.
                capture_age=max(0,age,depth_age)
                frame=CameraFrame(rgb,depth,dict(k),self.config["camera_to_base"].copy(),
                    Pose(0,0,0),self.monotonic()-capture_age,self.config["optical_frame"],stamp)
                frame.validate()
                self._last_stamp=stamp
                self.error=""
                return frame
            return None


def publication_payload(snapshot: dict, config: dict, *, ros_now: float,
                        monotonic_now: float) -> dict | None:
    """A failed/stale inference cannot masquerade as a fresh empty detection."""
    stamp,capture=snapshot.get("source_stamp"),snapshot.get("captured_at")
    if (snapshot.get("ready") is not True or not isinstance(stamp,(float,int))
            or not isinstance(capture,(float,int)) or not math.isfinite(stamp) or not math.isfinite(capture)
            or not -.05<=ros_now-stamp<=config["max_age"]
            or not 0<=monotonic_now-capture<=config["max_age"]):
        return None
    people=[]
    for person in snapshot.get("people",[]):
        x,y=person.get("x"),person.get("y")
        if not isinstance(x,(int,float)) or not isinstance(y,(int,float)) or not math.isfinite(x) or not math.isfinite(y):
            return None
        people.append({"x":x,"y":y,"confidence":person.get("confidence"),"id":person.get("id")})
    return {"stamp":stamp,"frame_id":config["base_frame"],"people":people,
            "source":"rgb:yolox-tiny+aligned-depth","optical_frame":config["optical_frame"],
            "inference_ms":snapshot.get("inference_ms")}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(argv)
    config=validate_config(json.loads(args.config.read_text()))
    try:
        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
        from sensor_msgs.msg import Image,CameraInfo
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError("Run the bridge in Dockerfile.ROS2 with ROS 2 rclpy/sensor_msgs/std_msgs") from error
    rclpy.init()
    node=rclpy.create_node("sassy_rosmaster_rgb_person_detector")
    node.set_parameters([Parameter("use_sim_time",value=config["use_sim_time"])])
    ros_now=lambda:node.get_clock().now().nanoseconds*1e-9
    source=RosCameraSource(config,ros_now)
    perception=CameraPerception(source,interval=config["inference_interval"]).start()
    qos=QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    publisher=node.create_publisher(String,config["people_topic"],qos)
    subscriptions=[]
    for kind,message_type in (("rgb",Image),("depth",Image),("camera_info",CameraInfo)):
        subscriptions.append(node.create_subscription(message_type,config[kind+"_topic"],
            lambda message,kind=kind:source.receive(kind,message),qos))
    last_stamp=-math.inf; last_warning=0.0
    def publish():
        nonlocal last_stamp,last_warning
        snapshot=perception.snapshot()
        payload=publication_payload(snapshot,config,ros_now=ros_now(),monotonic_now=time.monotonic())
        if payload is not None and payload["stamp"]>last_stamp:
            message=String(); message.data=json.dumps(payload,allow_nan=False)
            publisher.publish(message); last_stamp=payload["stamp"]
        elif payload is None and time.monotonic()-last_warning>5:
            reason=source.error or snapshot.get("error") or "Camera inference is not fresh enough for the hardware deadline"
            node.get_logger().warning(reason); last_warning=time.monotonic()
    timer=node.create_timer(.02,publish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        perception.close(); node.destroy_timer(timer); node.destroy_node(); rclpy.shutdown()


if __name__=="__main__":
    main()
