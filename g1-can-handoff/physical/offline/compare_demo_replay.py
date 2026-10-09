"""Compare the recorded failed demo with a physical harness replay, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compare(trial):
    path=Path(trial);value=json.loads(path.read_text());directory=Path(value['export']['directory']);fixture=json.loads((directory/'replay.json').read_text());source=Path(fixture['source']['local'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=fixture['source']['sha256']:raise ValueError('Original trace checksum mismatch')
    old=json.loads(source.read_text());rows=[json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()];rows=[r for r in rows if r['segment'].startswith('recorded-forward')]
    began=rows[0]['elapsed_s'];t=np.array([r['elapsed_s']-began for r in rows]);q=np.array([r['commanded_q'] for r in rows]);actual=np.array([r['measured_q'] for r in rows]);wire=np.array([r['sdk_reference']['q'][:7] for r in rows]);temp=np.array([r['temperatures_c'] for r in rows]);tau=np.array([r['measured_tau_est_nm'] for r in rows]);ot=np.array(fixture['time_s']);oq=np.array(fixture['left_q']);oa=np.array([r['measured'] for r in old['targets']]);duration=ot[-1]
    interpolated=np.array([np.interp(t,ot,oq[:,j]) for j in range(7)]).T
    parity=float(abs(q-interpolated).max())
    if parity>.001:raise ValueError(f'Recorded path/timing mismatch: {parity} rad')
    hold=t>=duration+.3
    if not hold.any():raise ValueError('Replay did not reach final target')
    result=dict(original_source=str(source),original_source_sha256=fixture['source']['sha256'],replay=str(path),same_original_timed_requested_path_with_20ms_interpolation=True,max_time_aligned_target_difference_rad=parity,forward_reached_final_target=True,reverse_return_completed=value['calibration']['completed'],original_final_error_rad=(np.array(old['failure']['goal_q'])-np.array(old['failure']['measured_q'])).tolist(),replay_hold_mean_error_rad=(q[hold]-actual[hold]).mean(0).tolist(),replay_hold_latest_sdk_goal_max_difference_rad=float(abs(wire[hold]-q[hold]).max()),peak_reported_shoulder_roll_torque_nm=float(abs(tau[:,1]).max()),last_shoulder_roll_casing_winding_c=temp[-1,1].tolist(),whole_trial_start_shoulder_roll_casing_winding_c=json.loads((directory/'samples.jsonl').read_text().splitlines()[0])['temperatures_c'][1],error=value.get('error'),control_released_verified=value['control_released_verified'],limitations=['Harness/support, contact environment, gains, current base pose and hand load differ from the recorded demo.','Encoder positions and reported torque; no independent wrist position or calibrated external force.','This comparison does not isolate gain effects from environmental restraint.'])
    fig,ax=plt.subplots(3,1,figsize=(11,8.5),sharex=True)
    fig.suptitle('Recorded reach failure reproduced on NX2 at P=75, D=3',fontsize=15)
    ax[0].plot(ot,oq[:,1],color='#2266aa',label='Original requested shoulder roll');ax[0].plot(ot,oa[:,1],color='#9c69a0',label='Original measured shoulder roll');ax[0].plot(t,actual[:,1],color='#d76b16',label='Harness replay measured');ax[0].plot(t,wire[:,1],color='#2266aa',ls='--',alpha=.6,label='Replay SDK reference');ax[0].set_ylabel('Position (rad)');ax[0].legend(ncol=2,fontsize=9)
    ax[1].plot(ot,abs(oq[:,1]-oa[:,1]),color='#9c69a0',label='Original shoulder error');ax[1].plot(t,abs(q[:,1]-actual[:,1]),color='#d76b16',label='Replay shoulder error');ax[1].set_ylabel('Error (rad)');ax[1].legend()
    ax[2].plot(t,temp[:,1,1],color='#d76b16',label='Shoulder winding');ax[2].plot(t,temp[:,1,0],color='#2266aa',label='Shoulder casing');ax[2].plot(t,temp[:,5,1],color='#775846',label='Wrist pitch winding');ax[2].axhline(65,ls='--',color='red',label='Experiment winding stop');ax[2].set_ylabel('Replay temperature (C)');ax[2].set_xlabel('Seconds into recorded forward trajectory');ax[2].legend(ncol=2)
    for a in ax:
        a.grid(alpha=.2);a.axvline(duration,color='grey',ls=':',alpha=.7)
    fig.text(.5,.01,'Same requested joint path/timing. Contact environment and gains differ; no fingers commanded.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.025,1,.97));out=path.parent/'plots';out.mkdir(exist_ok=True);figure=out/'recorded-demo-vs-harness.png';fig.savefig(figure,dpi=140);plt.close(fig)
    result['figure']=str(figure);(out/'recorded-demo-vs-harness.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trial');a=p.parse_args();print(json.dumps(compare(a.trial),indent=2))
