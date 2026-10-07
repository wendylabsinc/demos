"""HTTP gateway contract for radiometric cameras and acknowledged local valves."""
import json
import time
import urllib.request
import numpy as np

class HardwareAdapter:
    def __init__(self,config):
        self.sensor_url=config['thermal_url']; self.valve_url=config['valve_url']
        self.last_sequence=-1
    def frames(self,shape):
        with urllib.request.urlopen(self.sensor_url,timeout=2) as response: packet=json.load(response)
        age=time.time()-float(packet['timestamp'])
        if not 0<=age<2: raise ValueError('Stale or future thermal capture')
        seq=int(packet['sequence'])
        if seq<=self.last_sequence: raise ValueError('Repeated thermal capture sequence')
        values=np.asarray(packet['celsius'],dtype=float)
        if values.shape!=shape or not np.all(np.isfinite(values)) or values.min()<-40 or values.max()>1500:
            raise ValueError('Invalid radiometric frame dimensions or temperatures')
        self.last_sequence=seq
        return values
    def actuate(self,zone,on):
        body=json.dumps(dict(zone=zone,on=on)).encode()
        request=urllib.request.Request(self.valve_url,data=body,headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=2) as response: ack=json.load(response)
        if ack.get('zone')!=zone or ack.get('on') is not on or ack.get('ok') is not True:
            raise ValueError('Valve gateway did not acknowledge the requested state')
