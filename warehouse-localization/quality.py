"""Fail closed on invalid observations. No simulator or motion dependencies."""
import math
import numpy as np

def fresh(stamp, now, age=.8):
    return math.isfinite(stamp) and stamp>0 and -.05<=now-stamp<=age

def yaw(quaternion):
    x,y,z,w=quaternion
    if not all(math.isfinite(v) for v in quaternion) or abs(sum(v*v for v in quaternion)-1)>.02:
        raise ValueError('invalid orientation quaternion')
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def project_cloud(points, min_height=.15, max_height=1.5):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('invalid XYZ cloud')
    valid=np.all(np.isfinite(points),axis=1)&(points[:,2]>=min_height)&(points[:,2]<=max_height)
    points=points[valid]
    ranges=np.full(360,np.inf)
    distance=np.hypot(points[:,0],points[:,1])
    valid=(distance>=.35)&(distance<=12)
    points,distance=points[valid],distance[valid]
    bins=np.floor((np.arctan2(points[:,1],points[:,0])+math.pi)/(2*math.pi)*360).astype(int)%360
    np.minimum.at(ranges,bins,distance)
    if np.count_nonzero(np.isfinite(ranges))<60:raise ValueError('insufficient LiDAR coverage')
    return ranges

def pose_valid(covariance):
    c=np.asarray(covariance,float).reshape(6,6)
    planar=c[np.ix_([0,1,5],[0,1,5])]
    return (np.all(np.isfinite(c)) and np.allclose(planar,planar.T,atol=1e-6)
        and np.min(np.linalg.eigvalsh(planar))>=-1e-8
        and 0<=c[0,0]<=.0625 and 0<=c[1,1]<=.0625 and 0<=c[5,5]<=.0305)

def status(now, stamps, pose, mode):
    missing=[k for k in ('odom','scan') if not fresh(stamps.get(k,0),now)]
    ready=not missing and mode=='localization' and pose is not None and fresh(stamps.get('pose',0),now,2.)
    return dict(state='localized' if ready else 'mapping' if not missing and mode=='mapping' else 'waiting',
        ready=ready,missing=missing,pose=pose if ready else None,motion_enabled=False,
        mode=mode,frame='wendy_map',stamp=now)
