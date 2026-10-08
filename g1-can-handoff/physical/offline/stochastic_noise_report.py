"""Render the measured sim/real gaps and candidate-controller evidence."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(folder):
    validation=json.loads((folder/'held-out.json').read_text())
    controller=json.loads((folder/'controller-comparison.json').read_text())
    records=[r for r in validation['sessions'] if r['validated']]
    correlation={}
    for lag in ('.2s','1s','5s'):
        selected=[r for r in records if r['real']['correlation_over_time'][lag] is not None]
        correlation[lag]=dict(real_median=float(np.median([r['real']['correlation_over_time'][lag] for r in selected])),
                             generated_median=float(np.median([r['generated_correlation'][lag][1] for r in selected])))
    checks=[(r['real']['phase_boundary_stop_fraction'] if r['validated'] else r.get('real_phase_boundary_stop_fraction'),r.get('generated_phase_boundary_stop_probability')) for r in validation['sessions']]
    checks=[r for r in checks if all(v is not None for v in r)]
    summary=dict(sessions=len(validation['sessions']),evaluable_dense_sessions=len(records),
        held_out_p95_within_generated_90pct_interval=sum(r['p95_step_within_generated_90pct_interval'] for r in records),
        real_step_p95_range_mm=[min(r['real']['step_displacement_mm']['p95'] for r in records),max(r['real']['step_displacement_mm']['p95'] for r in records)],
        correlation=correlation,phase_checks=dict(sessions=len(checks),recorded_stop_fraction=float(np.mean([a for a,b in checks])),generated_stop_probability=float(np.mean([b for a,b in checks])),
            limitation='Only the observed XY phase checks; censored logs do not establish complete-demo stop probability.'),
        general_sensor_accuracy_calibrated=False,ground_truth_sessions=0,jump_samples=2,controller=controller)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axis=axes[0,0]
    actual=np.array([r['real']['step_displacement_mm']['p95'] for r in records])
    interval=np.array([r['generated_step_percentiles_mm']['p95'] for r in records])
    x=np.arange(len(records))
    axis.errorbar(x,interval[:,1],yerr=[interval[:,1]-interval[:,0],interval[:,2]-interval[:,1]],fmt='o',label='Fresh draws: 90% range',color='#1565c0')
    axis.scatter(x,actual,marker='x',label='Held-out real session',color='#222222',zorder=3)
    axis.set(title='Step displacement p95: session-held-out validation',ylabel='Millimetres',xlabel='Held-out session index');axis.legend(fontsize=8)
    axis=axes[0,1];labels=['0.2 s','1 s','5 s']
    axis.plot(labels,[correlation[k]['real_median'] for k in correlation],'o-',label='Real estimates',color='#222222')
    axis.plot(labels,[correlation[k]['generated_median'] for k in correlation],'x--',label='Stochastic estimates',color='#1565c0')
    axis.set(title='Time correlation: still mismatched at longer lags',ylabel='Group-demeaned correlation');axis.legend(fontsize=8)
    axis=axes[1,0]
    axis.bar(['Recorded phase samples','Generated sequences'],[summary['phase_checks']['recorded_stop_fraction']*100,summary['phase_checks']['generated_stop_probability']*100],color=['#555555','#1565c0'])
    axis.set(title='10 mm XY gate: censored phase samples',ylabel='Rejection (%)')
    axis=axes[1,1]
    axis.bar(['Baseline','Tuned candidate'],[controller['baseline_joint_change_rms_rad']*1000,controller['candidate_joint_change_rms_rad']*1000],color=['#555555','#16865e'])
    axis.set(title='IK target sensitivity: 24 held-out fresh draws',ylabel='Joint displacement RMS (mrad)')
    for axis in axes.flat:axis.grid(axis='y',alpha=.2)
    fig.suptitle('Stochastic Coke-position model and offline controller tuning\nCombined estimate variability; no independently measured can ground truth')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    print(json.dumps(report(parser.parse_args().directory),indent=2))
