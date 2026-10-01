"""Registration and persistence use captured camera frames, never scenario flags."""
import copy
import numpy as np
import pytest
from PIL import Image
from yard_drone.missions import MissionTracker,compare_images


def frame(image):
    return {'image':image,'depth':np.full(image.shape[:2],4.35,np.float32),'timestamp':1.,
        'cameraPosition':[0,0,4.35],'cameraRotation':np.eye(3).tolist(),
        'intrinsics':{'fx':188.,'fy':188.,'cx':159.5,'cy':119.5}}


def test_registration_persistence_and_no_ground_truth_labels(tmp_path):
    rng=np.random.default_rng(42)
    pixels=rng.integers(65,190,(240,320,3),dtype=np.uint8)
    f=frame(pixels)
    _,regions,_=compare_images(pixels,pixels,f,f)
    assert regions==[]
    tracker=MissionTracker(tmp_path)
    with pytest.raises(ValueError,match='baseline'):tracker.begin('current')
    tracker.begin('baseline')
    tracker.record(tracker.prepare('V00',f,[0,0,4.5],'A',tracker.mission['id']),95)
    tracker.finish();prior=tracker.baseline['id']
    restored=MissionTracker(tmp_path);assert restored.baseline['id']==prior
    restored.begin('current')
    changed=pixels.copy();changed[80:140,120:180]=[8,8,8]
    result=restored.prepare('V00',frame(changed),[0,0,4.5],'A',restored.mission['id'])
    restored.record(result,95);restored.finish()
    assert restored.incidents
    assert all(i['type']=='visual change' for i in restored.incidents)
    assert restored.baseline['id']==prior
    assert restored.incidents[0]['registration']['featureInliers']>=8
    assert (tmp_path/'evidence'/restored.incidents[0]['currentUrl'].rsplit('/',1)[-1]).exists()


def test_old_synthetic_benchmark_and_cancel_are_not_accepted(tmp_path):
    (tmp_path/'baseline.json').write_text('{"available":true}')
    t=MissionTracker(tmp_path);assert not t.baseline['available']
    t.begin('baseline');identifier=t.mission['id'];t.cancel()
    with pytest.raises(ValueError):t.finish()
    f=frame(np.zeros((240,320,3),np.uint8))
    assert not t.record(t.prepare('V00',f,[0,0,4.5],'A',identifier),95)
    assert not t.baseline['available']


def test_wrong_feature_warp_does_not_create_incident_on_unchanged_yard(monkeypatch):
    # Repeated construction edges can agree on a false translation. Even a
    # large feature consensus must not override identical measured poses.
    rng=np.random.default_rng(4)
    pixels=rng.integers(40,210,(240,320,3),dtype=np.uint8)
    pixels[20:220,18:25]=255
    wrong=np.array([[1.,0.,15.],[0.,1.,0.],[0.,0.,1.]])
    monkeypatch.setattr('cv2.findHomography',lambda *args:(wrong,np.ones((100,1),np.uint8)))
    _,regions,registration=compare_images(pixels,pixels,frame(pixels),frame(pixels))
    assert regions==[]
    assert registration['featureInliers']==100
    assert registration['method']=='calibrated ground-plane reprojection'
    assert 'disagrees' in registration['featureRejection']
    assert registration['calibrationDeviationPixels']['maximum']==pytest.approx(15.)


def test_disjoint_camera_views_have_no_comparable_pixels():
    pixels=np.zeros((240,320,3),np.uint8)
    previous=frame(pixels);current=frame(pixels)
    current['cameraPosition']=[15.,0.,4.35]
    _,regions,registration=compare_images(pixels,pixels,previous,current)
    assert regions==[]
    assert registration['validFraction']==0
    assert registration['comparison']=='No valid image overlap'
