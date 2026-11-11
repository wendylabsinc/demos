"""Compare one real-seeded simulated pickup and one recorded physical attempt.

Phase-progress comparisons quantify differences; they do not claim the two
runs had identical feedback/calibration or identical packet cadence.
"""
import argparse
import gzip
import json
from pathlib import Path
import tarfile
import numpy as np


def compare(simulation,trial):
    simulation=Path(simulation);trial=Path(trial)
    sim=json.loads(simulation.read_text());real=json.loads(trial.read_text())
    with tarfile.open(trial.with_suffix('.tar.gz')) as archive:
        events=[json.loads(r) for r in gzip.decompress(archive.extractfile('events.jsonl.gz').read()).splitlines()]
    targets=[r for r in events if r['phase']=='target'];phases=[]
    for phase in ('staging','approach','align','reach','lift'):
        a=[r for r in sim['trajectory'] if r['phase_name']==phase]
        b=[r for r in targets if r.get('phase_name')==phase]
        row=dict(phase=phase,sim_samples=len(a),real_samples=len(b))
        if a and b:
            # Each phase is aligned by normalized commanded progress, not time.
            progress=np.linspace(0,1,101)
            def resample(rows,key):
                values=np.asarray([r[key][:7] for r in rows])
                return np.array([np.interp(progress,np.linspace(0,1,len(rows)),values[:,i]) for i in range(7)]).T
            delta=resample(a,'arm')-resample(b,'arm')
            measured=resample(a,'measured_arm')-resample(b,'measured_arm')
            row.update(command_max_abs_difference_rad=float(np.max(np.abs(delta))),
                       command_rms_difference_rad=float(np.sqrt(np.mean(delta**2))),
                       measured_max_abs_difference_rad=float(np.max(np.abs(measured))),
                       real_max_tracking_error_rad=max(r['tracking_error_rad'] for r in b),
                       sim_max_tracking_error_rad=max(r['tracking_error_rad'] for r in a))
        phases.append(row)
    first_sim=next((r for r in sim['events'] if r['phase']=='plan'),None)
    plans=[r for r in events if r['phase']=='plan']
    initial=np.asarray(sim['setup_input']['measured']['body']['q'])
    current=np.asarray(real['controller_before']['measured']['body']['q'])
    return dict(schema='wendy.g1.sim-real-comparison.v1',simulation=str(simulation),trial=str(trial),
        simulation_lift_verified=sim['lift_verified'],physical_lift_verified=real.get('physical_lift_verified',False),
        simulation_error=sim.get('error'),real_error=real.get('error'),
        initial_body_state_difference_max_rad=float(np.max(np.abs(initial-current))),
        real_plans=len(plans),phase_progress_comparisons=phases,
        sim_elapsed_s=sim['simulated_time_s'],
        real_button_to_terminal_s=(real['finished_unix_ns']-real['started_unix_ns'])/1e9,
        sim_plan_waypoints=None if first_sim is None else first_sim['waypoints'],
        real_plan_waypoints=[p['waypoints'] for p in plans],
        planning_snapshots=sum(r['phase']=='planning_snapshot_complete' for r in events),
        limitations=['Real trial refreshes perception/calibration and replans after control handover.',
                     'Supported-base simulation does not reproduce factory balance or physical contact exactly.',
                     'A phase terminated early is a partial trajectory; its progress comparison is not full-path equivalence.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation',required=True);parser.add_argument('--trial',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=compare(args.simulation,args.trial)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
