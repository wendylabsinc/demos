"""Bounded physical staging/return test. No finger or can commands."""
import json,time,numpy as np,mujoco
from pathlib import Path
from runtime import Runtime,http_json
from physical_core import snapshot_guard,PhysicalPlant
from g1_can_handoff.sim import scene

def run(limits,table_height):
 r=Runtime()
 arm=None
 success=False;worst=0.
 try:
  time.sleep(.35)
  snap=snapshot_guard(r.read());archive=r.archive_guard()
  try:http_json(8090)
  except Exception:pass
  else:raise RuntimeError("Old onboard controller still running")
  if archive["age_s"].get("rt/arm_sdk") is not None and archive["age_s"]["rt/arm_sdk"]<2:raise RuntimeError("Arm publisher active")
  initial=np.array(snap["body"]["q"][15:29]);goal=initial.copy();goal[0]=.25;goal[1]=1.3;goal[3]=1.5
  # Check full joint-space path at a conservative table height, with measured fingers.
  p=scene(height=table_height+.05)
  plant=PhysicalPlant(p,None,None,r.read,{"arm_limits_rad":limits},r.event)
  poses=[initial+(goal-initial)*(t*t*(3-2*t)) for t in np.linspace(0,1,550)]
  p.ik_data.qpos[:]=p.data.qpos
  for q in poses:
   p.ik_data.qpos[p.qadr[p.arm]]=q[:7]
   mujoco.mj_forward(p.model,p.ik_data)
   if p.pose_collision(p.ik_data):raise RuntimeError("Staging path intersects conservative table/robot model")
  if np.any(goal<np.array(limits)[:,0]) or np.any(goal>np.array(limits)[:,1]):raise RuntimeError("Outside measured arm limits")
  from arm_reference import validate_motion_limits
  validate_motion_limits({"max_velocity_rad_s":.25,"max_acceleration_rad_s2":1.})
  from g1_arm import G1_29_ArmController
  r.event("physical_staging_start",initial=initial.tolist(),target=goal.tolist(),table_height_assumed_m=table_height,collision_table_margin_m=.05,operator_left_side_clear=True,hand_commands_enabled=False,recorder_session=archive["session"])
  arm=G1_29_ArmController(motion_mode=True,simulation_mode=False,lowstate_hub=r.hub,record_lowcmd=r.command_written)
  arm.set_arm_motion_limits({"max_velocity_rad_s":.25,"max_acceleration_rad_s2":1.})
  arm.set_arm_position_limits(limits)
  plant.adapter=arm
  def follow(path,phase):
   nonlocal worst
   r.event(phase)
   for i,q in enumerate(path):
    started=time.monotonic();fresh=plant.check()
    err=float(np.max(np.abs(np.array(fresh["body"]["q"][15:29])-q)));worst=max(worst,err)
    if err>.15:raise RuntimeError("Arm tracking exceeds 0.15 rad")
    arm.ctrl_dual_arm(q,np.zeros(14))
    r.event("target",phase_name=phase,arm=q.tolist(),tracking_error_rad=err)
    if i%50==0:r.archive_guard()
    time.sleep(max(0,.02-(time.monotonic()-started)))
  follow(poses,"staging_out")
  deadline=time.monotonic()+2
  while time.monotonic()<deadline:
   fresh=plant.check()
   if np.max(np.abs(np.array(fresh["body"]["q"][15:29])-goal))<.04:break
   time.sleep(.02)
  else:raise RuntimeError("Staging goal did not settle")
  r.event("staging_measured",measured_q=fresh["body"]["q"][15:29],physical_pickup_verified=False)
  follow(poses[::-1],"staging_return")
  time.sleep(.5);fresh=plant.check()
  if np.max(np.abs(np.array(fresh["body"]["q"][15:29])-initial))>.04:raise RuntimeError("Return did not settle")
  success=True
  return {"staging_and_return_verified":True,"worst_tracking_error_rad":worst,"physical_pickup_verified":False}
 except Exception as exc:
  r.event("physical_staging_error",error=str(exc),physical_pickup_verified=False)
  raise
 finally:
  if arm:
   arm.hold_arm_measured();arm.shutdown(ramp_s=1.)
   arm.msg.motor_cmd[29].q=0.;arm.msg.crc=arm.crc.Crc(arm.msg)
   result=arm.lowcmd_publisher.Write(arm.msg);r.command_written(None,result)
   r.event("staging_release",publisher_thread_stopped=not arm.publish_thread.is_alive(),motion_commands_sent=r.motion_commands_sent,automatic_home=False,dds_zero_weight_write=result,zero_weight_observed=None,success=success)
  r.hub.stop()
