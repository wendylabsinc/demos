"""Rebuild the measured October 8 paired comparison and standalone figure."""
import gzip,json,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/'validation/sim-real-20261008'
def read_events(name):
    with tarfile.open(ROOT/name/'trial-01.tar.gz') as tar:
        return [json.loads(line) for line in gzip.decompress(tar.extractfile('events.jsonl.gz').read()).splitlines()]
def cameras(rows,real):
    return [dict(phase=r['phase_name'],xy=r['geometry']['center_xy_m'],age_ms=r['exposure_age_ms'],fit_mm=r['geometry']['fit_error_m']*1000) for r in rows if r['phase']=='perception_result' and r.get('geometry')] if real else [dict(phase=r['phase_name'],xy=r['vision']['center_xy_m']) for r in rows if r['phase']=='camera']
real=read_events('real');retry=read_events('continuation')
sim=json.loads((ROOT/'simulation-current.json').read_text());held=json.loads((ROOT/'held-simulation.json').read_text())
replay=json.loads((ROOT/'real-planning-replay.json').read_text())
rc=cameras(real,True);sc=cameras(sim['events'],False);hc=cameras(retry,True);hs=cameras(held['events'],False)
realplans=[r for r in real if r['phase']=='plan']; simplans=[r for r in sim['events'] if r['phase']=='plan']
settle=next(r for r in real if r['phase']=='arm_settle_failed')
summary=dict(trials_stopped=True,robot_terminal_state='FROZEN',arms_held=True,
 first_pair=dict(sim_lift_verified=sim['lift_verified'],sim_seconds=sim['simulated_time_s'],real_lift_verified=False,
 real_seconds=json.loads((ROOT/'comparison.json').read_text())['real_button_to_terminal_s'],
 handover_estimated_can_shift_real_mm=float(np.linalg.norm(np.asarray(realplans[1]['waypoints'][2])-realplans[0]['waypoints'][2])*1000),
 handover_estimated_can_shift_sim_mm=float(np.linalg.norm(np.asarray(simplans[1]['waypoints'][2])-simplans[0]['waypoints'][2])*1000),
 reach_settle_error_real_rad=max(abs(v) for v in settle['error_rad']),settle_limit_rad=.04,
 reach_tracking_max_sim_rad=max(r['tracking_error_rad'] for r in sim['trajectory'] if r['phase_name']=='reach'),
 reach_tracking_max_real_rad=max(r['tracking_error_rad'] for r in real if r['phase']=='target' and r.get('phase_name')=='reach')),
 continuation=dict(sim_lift_verified=held['lift_verified'],sim_seconds=held['simulated_time_s'],real_lift_verified=False,
 estimated_target_shift_mm=float(np.linalg.norm(np.asarray(hc[-1]['xy'])-hc[0]['xy'])*1000),
 target_shift_limit_mm=10.,failure_phase=hc[-1]['phase'],exposure_age_ms=hc[-1]['age_ms'],cylinder_fit_error_mm=hc[-1]['fit_mm']),
 exact_input_ik_replay=dict(decisions=replay['decisions'],bitwise_identical=sum(r['identical'] for r in replay['results']),
 maximum_joint_difference_rad=max(r['max_abs_error_rad'] or 0. for r in replay['results']),
 same_sources_models_and_package_versions=True,numerical_platform_equivalence_not_proven=True),
 real_camera_measurements_first=rc,real_camera_measurements_continuation=hc,
 implications=['IK replay differences are negligible relative to observed tracking and input changes.',
 'Target positions are estimates: recordings do not independently establish physical can motion versus camera/tracker error.',
 'Simulation holds the base/camera steady and supplies ideal rendered RGB-D; real balance, translation, extrinsic flex and partial-view fitting remain gaps.',
 'The real settled shoulder-roll error is larger than simulation predicts; gain/feedforward, loading and adapter behavior need a measured-state replay.',
 'Timing compares simulated seconds against physical wall-clock seconds, including real planning and handover.'])
(ROOT/'gap-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
fig.suptitle('Real-seeded simulation vs physical G1: where they diverged',fontsize=17,fontweight='bold')
ax=axes[0,0]
for label,rows,timekey,colour in [('Simulation',sim['trajectory'],'simulation_time_s','#148778'),('Real',[r for r in real if r['phase']=='target'],'monotonic_ns','#d45c35')]:
    rows=[r for r in rows if r.get('phase_name')=='reach'];x=np.linspace(0,100,len(rows))
    ax.plot(x,[r['arm'][1] for r in rows],color=colour,label=label+' command',linewidth=2)
    ax.plot(x,[r['measured_arm'][1] for r in rows],color=colour,linestyle='--',label=label+' measured',linewidth=1.7)
ax.set(title='1. Shoulder roll during reach',xlabel='Commanded phase progress (%)',ylabel='Joint angle (rad)')
ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
ax.text(.02,.03,'Different refreshed target inputs; not an identical command replay.',transform=ax.transAxes,fontsize=8,color='#475569')
ax=axes[0,1]
labels=['Sim: reach\ntracking max','Real: reach\ntracking max','Real: after\n2 s settling']
values=[summary['first_pair']['reach_tracking_max_sim_rad'],summary['first_pair']['reach_tracking_max_real_rad'],summary['first_pair']['reach_settle_error_real_rad']]
bars=ax.bar(labels,values,color=['#148778','#d45c35','#b84229'],width=.55)
ax.axhline(.04,color='#52647b',linestyle='--',label='Final settling limit: 0.04 rad')
for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.001,f'{v:.5f}',ha='center')
ax.set(title='2. The physical arm settles less accurately',ylabel='Maximum joint error (rad)',ylim=(0,.055));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
ax=axes[1,0]
for label,rows,colour in [('Real retry',hc,'#d45c35'),('Sim from held pose',hs,'#148778')]:
    coords=np.asarray([r['xy'] for r in rows]);displacement=np.linalg.norm(coords-coords[0],axis=1)*1000
    # Pre-lift checkpoints only, using the same phase labels across both runs.
    phases=[r['phase'] for r in rows];wanted=['planning','plan-refresh','pre-open','pre-stage','approach','align','reach']
    selected=[(wanted.index(p),v) for p,v in zip(phases,displacement) if p in wanted]
    ax.plot([p for p,v in selected],[v for p,v in selected],'-o',color=colour,label=label,linewidth=2,markersize=4)
ax.axhline(10,color='#52647b',linestyle='--',label='Pregrasp shift limit: 10 mm')
ax.set_xticks(range(7),['Plan','Refresh','Open','Stage','Approach','Align','Reach'],rotation=25)
ax.set(title='3. Estimated can position drifts in real views',ylabel='Displacement from planning estimate (mm)',ylim=(-.5,14));ax.legend(fontsize=8);ax.grid(alpha=.15)
ax.text(.02,.96,'Failure frame: 246 ms old; depth fit error 3.13 mm',transform=ax.transAxes,fontsize=9,va='top')
ax=axes[1,1]
differences=np.sort([r['max_abs_error_rad'] or 0 for r in replay['results']])*1e6
ax.plot(np.arange(1,len(differences)+1),differences,color='#496aa8',linewidth=2)
ax.set(title='4. Exact-input replay: 114 real IK decisions',xlabel='Decisions sorted by numerical difference',ylabel='Largest joint difference (microradians)')
ax.grid(alpha=.15);ax.text(.03,.95,f'Max = {max(differences):.2f} microradians\nNot bitwise identical across execution platforms',transform=ax.transAxes,va='top',fontsize=10)
fig.savefig(ROOT/'gap.png',dpi=150)
print(json.dumps({k:v for k,v in summary.items() if k not in ('real_camera_measurements_first','real_camera_measurements_continuation')},indent=2))
