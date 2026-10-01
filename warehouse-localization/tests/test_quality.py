import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from quality import fresh,project_cloud,pose_valid,status

def test_missing_and_stale_inputs_never_return_origin_as_pose():
    assert status(10,{},None,'localization')['pose'] is None
    p={'x':1,'y':2,'yaw':0}
    assert not status(10,{'odom':9,'scan':10,'pose':10},p,'localization')['ready']
    assert status(10,{'odom':10,'scan':10,'pose':10},p,'localization')['ready']
    assert not fresh(100,10) and not fresh(0,10)

def test_cloud_projects_nearest_obstacle_and_rejects_sparse_or_ground_data():
    a=np.linspace(-np.pi,np.pi,360,endpoint=False)
    pts=np.c_[3*np.cos(a),3*np.sin(a),np.ones(360)]
    ranges=project_cloud(pts)
    assert np.isclose(np.min(ranges),3)
    import pytest
    with pytest.raises(ValueError):project_cloud(pts[:10])
    pts[:,2]=0
    with pytest.raises(ValueError):project_cloud(pts)

def test_high_uncertainty_and_nonpositive_covariance_rejected():
    c=np.zeros((6,6));c[0,0]=c[1,1]=.01;c[5,5]=.005
    assert pose_valid(c)
    c[0,0]=1;assert not pose_valid(c)
    c[0,0]=-.01;assert not pose_valid(c)
