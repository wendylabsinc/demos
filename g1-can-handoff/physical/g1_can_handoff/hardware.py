"""Read-only G1/Dex3 commissioning probe. It never creates command publishers."""
import argparse
import json
import time
import threading
import numpy as np

def probe(interface,domain,seconds=3.):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_,HandState_
    ChannelFactoryInitialize(domain,interface)
    lock=threading.Lock();samples={"body":[],"dex3":[]}
    def callback(name):
        def receive(message):
            with lock:
                samples[name].append((time.monotonic(),message))
                samples[name]=samples[name][-2000:]
        return receive
    subscriptions=[]
    for name,topic,message in (("body","rt/lowstate",LowState_),("dex3","rt/dex3/left/state",HandState_)):
        sub=ChannelSubscriber(topic,message);sub.Init(callback(name),10);subscriptions.append(sub)
    time.sleep(seconds)
    result={"motion_enabled":False,"interface":interface,"domain":domain}
    with lock:
        for name,values in samples.items():
            if len(values)<2:raise RuntimeError(f"No live {name} telemetry on the selected interface/domain")
            times=[t for t,_ in values]
            age=time.monotonic()-times[-1]
            if age>.25:raise RuntimeError(f"Stale {name} telemetry")
            motors=values[-1][1].motor_state
            q=np.array([motor.q for motor in motors],dtype=float)
            if not np.all(np.isfinite(q)):raise RuntimeError(f"Invalid {name} joint state")
            result[name]={"samples":len(values),"rate_hz":(len(values)-1)/(times[-1]-times[0]),"age_s":age,"joint_positions":q.tolist()}
        if len(result["body"]["joint_positions"])<29 or len(result["dex3"]["joint_positions"])!=7:
            raise RuntimeError("Telemetry does not match a G1 body with a seven-motor Dex3")
    return result

def main():
    parser=argparse.ArgumentParser(description="Observe G1 and left Dex3 telemetry without publishing motion")
    parser.add_argument("--interface",required=True)
    parser.add_argument("--domain",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(probe(args.interface,args.domain),indent=2))

if __name__=="__main__":main()
