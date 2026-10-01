"""Capture-time synchronization and calibration tests, without a ROS runtime."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

from sassy_rosmaster.perception import ground_people
from sassy_rosmaster.perception_ros import (RosCameraSource,camera_intrinsics,decode_image,
    publication_payload,stamp_seconds,validate_config)


def config():
    return {"calibration_verified":True,"rgb_rectified":True,"depth_registered_to_rgb":True,
        "fixed_camera_mount":True,"rgb_topic":"/camera/rgb_rectified","depth_topic":"/camera/aligned_depth",
        "camera_info_topic":"/camera/camera_info","people_topic":"/sassy_rosmaster/people",
        "optical_frame":"camera_optical","base_frame":"base_link",
        "camera_to_base":[[0,0,1,.2],[-1,0,0,0],[0,-1,0,.33],[0,0,0,1]],
        "camera_info_static":False,"use_sim_time":False,"sync_slop":.025,"max_age":.3,"inference_interval":.1}


def header(stamp=10.,frame="camera_optical"):
    sec=int(stamp)
    return NS(stamp=NS(sec=sec,nanosec=round((stamp-sec)*1e9)),frame_id=frame)


def rgb(stamp=10.,frame="camera_optical",encoding="rgb8"):
    row=bytes([10,20,30])*16+b"pad!"
    return NS(header=header(stamp,frame),width=16,height=16,step=52,data=row*16,
              encoding=encoding,is_bigendian=False)


def depth(stamp=10.,big=False):
    row=np.full(16,2000,dtype=">u2" if big else "<u2").tobytes()+b"xx"
    return NS(header=header(stamp),width=16,height=16,step=34,data=row*16,
              encoding="16UC1",is_bigendian=big)


def info(stamp=10.):
    return NS(header=header(stamp),width=16,height=16,
        p=[20,0,7.5,0,0,20,7.5,0,0,0,1,0],r=[1,0,0,0,1,0,0,0,1],
        binning_x=0,binning_y=0,roi=NS(x_offset=0,y_offset=0,width=0,height=0))


class CameraBridgeTests(unittest.TestCase):
    def test_example_refuses_guessed_calibration(self):
        example=Path(__file__).resolve().parents[1]/"perception.example.json"
        with self.assertRaisesRegex(ValueError,"calibration_verified"):
            validate_config(json.loads(example.read_text()))
        c=config();c["camera_to_base"]=None
        with self.assertRaisesRegex(ValueError,"camera_to_base"):
            validate_config(c)

    def test_fixed_rigid_optical_transform_required(self):
        for key,value in (("rgb_rectified",False),("depth_registered_to_rgb",False),
                          ("fixed_camera_mount",False),("rgb_topic","relative"),
                          ("max_age",.4),("sync_slop",.2)):
            c=config();c[key]=value
            with self.assertRaises(ValueError,msg=key):validate_config(c)
        c=config();c["camera_to_base"][0][2]=2
        with self.assertRaises(ValueError):validate_config(c)

    def test_rgb_padding_and_bgr_channel_order(self):
        image=decode_image(rgb(encoding="bgr8"))
        self.assertEqual(image.shape,(16,16,3))
        self.assertEqual(list(image[0,0]),[30,20,10])
        self.assertEqual(list(image[-1,-1]),[30,20,10])

    def test_padded_depth_millimetres_and_endianness(self):
        for big in (False,True):
            image=decode_image(depth(big=big))
            self.assertEqual(image.shape,(16,16))
            self.assertTrue(np.all(image==2.))
        message=depth();message.data=message.data[:-1]
        with self.assertRaisesRegex(ValueError,"stride"):decode_image(message)

    def test_unsupported_calibration_and_missing_stamp_rejected(self):
        for change in (lambda m:m.p.__setitem__(3,2),lambda m:m.r.__setitem__(0,0),
                       lambda m:setattr(m,"binning_x",2),lambda m:setattr(m.roi,"x_offset",1)):
            message=info();change(message)
            with self.assertRaises(ValueError):camera_intrinsics(message)
        with self.assertRaises(ValueError):stamp_seconds(header(0))
        self.assertEqual(stamp_seconds(header(0),allow_zero=True),0)

    def test_capture_stamp_preserved_and_depth_grounded_to_base(self):
        source=RosCameraSource(config(),lambda:10.1,monotonic=lambda:100.)
        for kind,message in (("rgb",rgb()),("depth",depth()),("camera_info",info())):
            source.receive(kind,message)
        frame=source()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.source_stamp,10.)
        self.assertAlmostEqual(frame.captured_at,99.9)
        people=ground_people(frame,[{"label":"person","confidence":.9,"bbox":[0,0,15,15]}])
        self.assertEqual(len(people),1)
        self.assertAlmostEqual(people[0]["x"],2.2)
        self.assertAlmostEqual(people[0]["y"],0.)
        self.assertEqual(people[0]["source"],"rgb:yolox-tiny+aligned-depth")
        self.assertIsNone(source(),"The same camera frame must not be processed repeatedly")

    def test_unsynchronized_and_wrong_frame_pixels_never_form_observation(self):
        source=RosCameraSource(config(),lambda:10.1,monotonic=lambda:100.)
        source.receive("rgb",rgb())
        source.receive("depth",depth(10.08))
        source.receive("camera_info",info())
        self.assertIsNone(source())
        self.assertIn("synchronized",source.error)
        source.receive("depth",depth())
        source.receive("rgb",rgb(frame="wrong_optical"))
        self.assertIsNone(source())
        self.assertIn("optical_frame",source.error)

    def test_freshness_uses_oldest_pixel_capture_and_preserves_rgb_stamp(self):
        source=RosCameraSource(config(),lambda:10.1,monotonic=lambda:100.)
        for kind,message in (("rgb",rgb(10.)),("depth",depth(9.98)),("camera_info",info(10.))):
            source.receive(kind,message)
        frame=source()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.source_stamp,10.,"Outgoing ROS stamp must remain the RGB capture stamp")
        self.assertAlmostEqual(frame.captured_at,99.88,"Depth's older capture must bound freshness")
        snapshot={"ready":True,"source_stamp":frame.source_stamp,"captured_at":frame.captured_at,"people":[]}
        self.assertIsNone(publication_payload(snapshot,config(),ros_now=10.29,monotonic_now=100.19),
                          "Inference must expire on depth age even when RGB is still fresh")

    def test_selected_depth_expiring_after_receipt_rejects_frame_delivery(self):
        now=[10.01]
        source=RosCameraSource(config(),lambda:now[0],monotonic=lambda:100.)
        for kind,message in (("rgb",rgb(9.73)),("depth",depth(9.71)),("camera_info",info(9.73))):
            source.receive(kind,message)
        now[0]=10.02  # RGB age .29, aligned depth age .31 within .025 sync slop.
        self.assertIsNone(source())
        self.assertIn("Depth capture expired",source.error)

    def test_missing_or_invalid_depth_does_not_invent_people_position(self):
        source=RosCameraSource(config(),lambda:10.1,monotonic=lambda:100.)
        empty=depth();empty.data=b"\0"*empty.step*empty.height
        for kind,message in (("rgb",rgb()),("depth",empty),("camera_info",info())):
            source.receive(kind,message)
        frame=source()
        self.assertEqual(ground_people(frame,[{"label":"person","confidence":.99,"bbox":[0,0,15,15]}]),[])

    def test_static_calibration_requires_explicit_contract(self):
        c=config();c["camera_info_static"]=True
        source=RosCameraSource(c,lambda:10.1,monotonic=lambda:100.)
        for kind,message in (("rgb",rgb()),("depth",depth()),("camera_info",info(0))):
            source.receive(kind,message)
        self.assertIsNotNone(source())
        source2=RosCameraSource(config(),lambda:10.1)
        source2.receive("camera_info",info(0))
        self.assertIn("missing",source2.error)

    def test_failed_or_expired_inference_cannot_publish_empty_fresh_heartbeat(self):
        snapshot={"ready":True,"source_stamp":10.,"captured_at":99.9,"people":[],"inference_ms":40}
        payload=publication_payload(snapshot,config(),ros_now=10.15,monotonic_now=100.05)
        self.assertEqual(payload["stamp"],10.)
        self.assertEqual(payload["frame_id"],"base_link")
        self.assertEqual(payload["people"],[])
        self.assertIsNone(publication_payload(snapshot,config(),ros_now=10.31,monotonic_now=100.1))
        self.assertIsNone(publication_payload(snapshot,config(),ros_now=10.15,monotonic_now=100.21))
        self.assertIsNone(publication_payload({**snapshot,"ready":False},config(),ros_now=10.15,monotonic_now=100.05))


if __name__=="__main__":
    unittest.main()
