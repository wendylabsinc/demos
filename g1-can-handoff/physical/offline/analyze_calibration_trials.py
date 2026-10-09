"""Plot recorded encoder response and temperatures; never contacts hardware."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES=['Shoulder pitch','Shoulder roll','Shoulder yaw','Elbow','Wrist roll','Wrist pitch','Wrist yaw']

def analyze(path):
    path=Path(path);value=json.loads(path.read_text());export=value['export']
    archive=Path(export['archive'])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=export['sha256']:
        raise ValueError('Evidence checksum mismatch')
    directory=Path(export['directory']);rows=[json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()]
    t=np.array([r['elapsed_s'] for r in rows]);q=np.array([r['commanded_q'] for r in rows]);actual=np.array([r['measured_q'] for r in rows]);wire=np.array([r['sdk_reference']['q'][:7] for r in rows]);temp=np.array([r['temperatures_c'] for r in rows]);tau=np.array([r['measured_tau_est_nm'] for r in rows])
    if not len(rows) or not np.all(np.diff(t)>0):raise ValueError('Incomplete or unordered samples')
    out=path.parent/'plots';out.mkdir(exist_ok=True);stem=path.stem
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    fig.suptitle(f'{stem}: real encoder tracking and motor temperatures',fontsize=14)
    axes[0].plot(t,q[:,1],label='Requested shoulder roll',lw=2);axes[0].plot(t,wire[:,1],label='Latest SDK reference',ls='--');axes[0].plot(t,actual[:,1],label='Measured shoulder roll');axes[0].set_ylabel('Joint position (rad)');axes[0].legend(ncol=3)
    for j,name in enumerate(NAMES):axes[1].plot(t,abs(q[:,j]-actual[:,j]),label=name,alpha=.9)
    axes[1].axhline(.15,color='red',ls='--',label='Experiment stop');axes[1].set_ylabel('Absolute error (rad)');axes[1].legend(ncol=4,fontsize=8)
    for j,name in enumerate(NAMES):
        axes[2].plot(t,temp[:,j,1],label=name+' winding');axes[3].plot(t,tau[:,j],label=name)
    axes[2].plot(t,temp[:,1,0],ls='--',color='black',label='Shoulder roll casing');axes[2].set_ylabel('Temperature (C)');axes[2].legend(ncol=4,fontsize=8)
    axes[3].set_ylabel('Reported torque (Nm)');axes[3].set_xlabel('Seconds since arm acquisition');axes[3].legend(ncol=4,fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        for i in range(1,len(rows)):
            if rows[i]['segment']!=rows[i-1]['segment']:ax.axvline(t[i],color='grey',alpha=.2)
    fig.tight_layout();figure=out/f'{stem}.png';fig.savefig(figure,dpi=140);plt.close(fig)
    summary=dict(trial=str(path),completed=value['calibration']['completed'],error=value.get('error'),control_released=value.get('control_released_verified'),samples=len(rows),duration_s=float(t[-1]),request_rms_error_rad=np.sqrt(np.mean((q-actual)**2,axis=0)).tolist(),latest_reference_rms_error_rad=np.sqrt(np.mean((wire-actual)**2,axis=0)).tolist(),peak_reported_torque_nm=abs(tau).max(axis=0).tolist(),start_temperatures_c=temp[0].tolist(),end_temperatures_c=temp[-1].tolist(),peak_temperatures_c=temp.max(axis=0).tolist(),figure=str(figure),independent_cartesian_ground_truth=False,temperature_resolution_c=1)
    phases={}
    for phase in dict.fromkeys(r['segment'] for r in rows):
        mask=np.array([r['segment']==phase for r in rows]);phases[phase]=dict(samples=int(mask.sum()),shoulder_roll_mean_error_rad=float((q-actual)[mask,1].mean()),shoulder_roll_rms_error_rad=float(np.sqrt(np.mean((q-actual)[mask,1]**2))),peak_joint_error_rad=float(abs(q-actual)[mask].max()),peak_shoulder_roll_torque_nm=float(abs(tau[mask,1]).max()),temperatures_end_c=temp[mask][-1].tolist())
    summary['phases']=phases
    (out/f'{stem}.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/f'{stem}.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seconds','segment','joint','requested_rad','latest_sdk_reference_rad','measured_rad','reported_torque_nm','casing_c','winding_c'])
        for i,r in enumerate(rows):
            for j,n in enumerate(NAMES):writer.writerow([t[i],r['segment'],n,q[i,j],wire[i,j],actual[i,j],tau[i,j],*temp[i,j]])
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trial',nargs='+');a=p.parse_args()
    for trial in a.trial:print(json.dumps(analyze(trial),indent=2))
