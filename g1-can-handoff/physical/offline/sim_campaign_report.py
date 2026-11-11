"""Verify archived episodes and summarize a completed paired simulation campaign."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def family(error):
    text=error.lower()
    for words,label in ((('worker','wall timeout'),'Test infrastructure'),(('collision','intersects','reachable','orientation search'),'Planning/collision'),
                        (('moved','tracking bound'),'Estimated position movement'),(('absent','ambiguous','cylinder','depth','upright'),'Perception/geometry'),
                        (('stale','exposure'),'Freshness'),(('settle','tracking error'),'Control tracking'),(('lift','obstruction','grasp'),'Grasp/lift')):
        if any(word in text for word in words):return label
    return 'Other'


def report(folder):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text());rows=json.loads((folder/'outcomes.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    expected={(case['id'],policy) for case in manifest['cases'] for policy in ('legacy','tuned')}
    actual={(row['case_id'],row['policy']) for row in rows}
    if actual!=expected or len(actual)!=len(rows):raise ValueError('Campaign incomplete or duplicate outcomes')
    hashes={};sources=None;family_counts={policy:Counter() for policy in ('legacy','tuned')};failure_details=[]
    for row in rows:
        if row.get('episode'):
            path=folder/row['episode'];packed=path.read_bytes();digest=hashlib.sha256(packed).hexdigest()
            if digest!=row['episode_sha256']:raise ValueError('Episode checksum mismatch: '+str(path))
            episode=json.loads(gzip.decompress(packed));hashes[row['episode']]=digest
            if episode.get('physical_pickup_verified') or not episode.get('simulation_only'):raise ValueError('Unexpected physical result in sim campaign')
            if episode['lift_verified']!=row['lift_verified']:raise ValueError('Episode/summary outcome mismatch')
            if episode['perception']['yolo_calls']!=0:raise ValueError('Unexpected classifier call')
            if sources is None:sources=episode['source_sha256']
            if sources!=episode['source_sha256']:raise ValueError('Runtime source changed during campaign')
            if not row['lift_verified']:
                collisions=[event['collision'] for event in episode['events'] if event.get('phase')=='finger_trajectory_collision']
                failure_details.append(dict(case=row['case_id'],policy=row['policy'],offset_xy_m=row['offset_xy_m'],error=row['error'],
                    command_count=episode['command_count'],commands_at_failure=episode.get('commands_at_failure'),
                    first_finger_collision=collisions[0] if collisions else None))
        if not row['lift_verified']:family_counts[row['policy']][family(row['error'])]+=1
    pairs={}
    for row in rows:pairs.setdefault(row['case_id'],{})[row['policy']]=row
    per_group={};execution_cost={}
    for condition in summary['groups']:
        selected=[p for p in pairs.values() if p['legacy']['condition']==condition]
        counts=Counter('both_pass' if p['legacy']['lift_verified'] and p['tuned']['lift_verified'] else 'tuned_only' if p['tuned']['lift_verified'] else 'legacy_only' if p['legacy']['lift_verified'] else 'both_fail' for p in selected)
        discordant=counts['tuned_only']+counts['legacy_only']
        pvalue=min(1.,2*sum(math.comb(discordant,k) for k in range(min(counts['tuned_only'],counts['legacy_only'])+1))/2**discordant) if discordant else 1.
        per_group[condition]=dict(outcomes=dict(counts),exact_paired_sign_test_p=pvalue)
        execution_cost[condition]={policy:dict(mean_episode_wall_time_s=float(np.mean([r['wall_time_s'] for r in rows if r['condition']==condition and r['policy']==policy and not r.get('harness_error')]))) for policy in ('legacy','tuned')}
    extra=dict(episodes_verified=len(hashes),source_sha256=sources,per_condition_paired=per_group,
        failure_families={p:dict(c) for p,c in family_counts.items()},failure_details=failure_details,execution_cost=execution_cost,all_raw_episode_hashes=hashes,
        hardware_used=False,physical_pickup_verified=False)
    (folder/'analysis.json').write_text(json.dumps(extra,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    labels={'fitted_noise':'Fitted noise\n32 cases','nearby_placement':'Nearby placement\n16 cases','double_noise':'Double noise\n16 cases'}
    colors={'legacy':'#555555','tuned':'#16865e'}
    axis=axes[0,0];conditions=('fitted_noise','nearby_placement','double_noise');x=np.arange(len(conditions))
    for offset,policy in ((-.18,'legacy'),(.18,'tuned')):
        rates=np.array([summary['groups'][c][policy]['lift_fraction'] for c in conditions]);ci=np.array([summary['groups'][c][policy]['lift_fraction_wilson95'] for c in conditions])
        axis.bar(x+offset,rates*100,width=.34,color=colors[policy],label=policy.capitalize())
        axis.errorbar(x+offset,rates*100,yerr=np.array([np.maximum(0,rates-ci[:,0]),np.maximum(0,ci[:,1]-rates)])*100,fmt='none',color='#222',capsize=3)
    axis.set_xticks(x,[labels[c] for c in conditions]);axis.set(title='Verified simulated lift; conditional 95% intervals',ylabel='Completion (%)',ylim=(0,110));axis.legend()
    axis=axes[0,1]
    passed=[p for p in pairs.values() if all(r['lift_verified'] for r in p.values())]
    if passed:
        a=np.array([p['legacy']['simulated_time_s'] for p in passed]);b=np.array([p['tuned']['simulated_time_s'] for p in passed])
        axis.scatter(a,b,s=22,color=colors['tuned']);lo=min(a.min(),b.min())-.2;hi=max(a.max(),b.max())+.2
        axis.plot([lo,hi],[lo,hi],'--',color='#999')
    axis.set(title=f'Paired duration: both-pass cases only (n={len(passed)})',xlabel='Legacy simulated seconds',ylabel='Tuned simulated seconds')
    axis=axes[1,0];categories=sorted(set(family_counts['legacy'])|set(family_counts['tuned']))
    if categories:
        x=np.arange(len(categories))
        for offset,policy in ((-.18,'legacy'),(.18,'tuned')):axis.bar(x+offset,[family_counts[policy][c] for c in categories],width=.34,color=colors[policy],label=policy.capitalize())
        axis.set_xticks(x,categories,rotation=15,ha='right');axis.legend()
    else:axis.text(.5,.5,'No failed episodes',ha='center',va='center',transform=axis.transAxes)
    axis.set(title='Preserved failure families',ylabel='Episodes')
    axis=axes[1,1]
    for policy in ('legacy','tuned'):
        values=np.sort([r['max_tracking_error_rad'] for r in rows if r['policy']==policy and r['lift_verified']])
        if len(values):axis.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=policy.capitalize(),color=colors[policy])
    axis.set(title='Maximum tracking error: successful episodes only',xlabel='Radians',ylabel='Cumulative fraction');axis.legend()
    for axis in axes.flat:axis.grid(axis='y',alpha=.2)
    fig.suptitle('128 full offline simulations: frozen legacy and tuned planners\nOne captured robot setup; physical servo/perception gaps remain')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    return dict(completed=len(rows),verified=len(hashes),failure_families=extra['failure_families'],paired=summary['paired_outcomes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    print(json.dumps(report(parser.parse_args().directory),indent=2))
