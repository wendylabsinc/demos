"""Optional onboard speaker output. Browser audio needs no system binaries."""
import array
import math
import os
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor

class RobotAudio:
    def __init__(self):
        self.enabled=os.getenv('ROBOT_AUDIO','0')=='1'
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rover-voice')
    def emit(self,kind,text):
        if self.enabled and kind in ('speech','horn'):
            self.pool.submit(self._play,kind,text)
    def _play(self,kind,text):
        try:
            if kind=='speech' and shutil.which('espeak-ng'):
                subprocess.run(['espeak-ng','-v','en-gb','-s','170',text],timeout=12,check=False)
            elif kind=='horn' and shutil.which('aplay'):
                rate=22050
                samples=array.array('h')
                for i in range(int(rate*.65)):
                    t=i/rate
                    on=t<.22 or .3<t<.56
                    value=.2*(math.sin(2*math.pi*420*t)+math.sin(2*math.pi*510*t)) if on else 0
                    samples.append(int(32767*value))
                with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                    with wave.open(f.name,'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(rate)
                        wav.writeframes(samples.tobytes())
                    subprocess.run(['aplay','-q',f.name],timeout=3,check=False)
        except (OSError,subprocess.TimeoutExpired):
            pass
    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)
