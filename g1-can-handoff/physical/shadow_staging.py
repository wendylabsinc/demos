"""Search a measured joint-space staging pose; collision checks only."""
import json,numpy as np,urllib.request,mujoco
from g1_can_handoff.sim import scene
from physical_core import PhysicalPlant
def staging(height,limits):
 p=scene(height=height)
 plant=PhysicalPlant(p,None,None,lambda:json.load(urllib.request.urlopen("http://127.0.0.1:8102/status"))["measured"],{"arm_limits_rad":limits},lambda *a,**k:None)
 start=p.data.qpos[p.qadr[p.arm]].copy()
 candidates=[]
 for pitch in [.25,0,-.2,-.5,-.8,-1.2,-1.5]:
  for roll in [.28,.5,.8,1.1,1.3,1.5]:
   for elbow in [1.2,1.5,1.8]:
    goal=start.copy();goal[0]=pitch;goal[1]=roll;goal[3]=elbow
    if np.any(goal<np.array(limits)[:7,0]) or np.any(goal>np.array(limits)[:7,1]):continue
    p.ik_data.qpos[:]=p.data.qpos
    ok=True
    for t in np.linspace(0,1,200):
     p.ik_data.qpos[p.qadr[p.arm]]=start+(goal-start)*(t*t*(3-2*t))
     mujoco.mj_forward(p.model,p.ik_data)
     if p.pose_collision(p.ik_data):ok=False;break
    site=p.ik_data.site_xpos[p.site].copy()
    if ok and site[0]<.20 and site[2]>height+.16:
     candidates.append({"joint_goal":goal.tolist(),"site":site.tolist(),"travel_rad":float(np.max(np.abs(goal-start)))})
 return {"actuation":False,"estimated_geometry_only":True,"candidates":sorted(candidates,key=lambda c:c["travel_rad"])}
