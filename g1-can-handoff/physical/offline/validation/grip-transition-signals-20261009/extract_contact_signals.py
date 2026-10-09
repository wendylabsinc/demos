import json,gzip,hashlib,pathlib,statistics
from export_cdr import decode
cases=[
 {'name':'earlier-211444','origin':1791580484057222918,'stop':1791580489578870162,'file':'/records/20261009T210904Z-e2785e7e/samples-000005.jsonl.gz','sha':'33e80758e2b03a20ed07ef11189e20804d4fc7b509ac04cfed8072afc7d6551f'},
 {'name':'latest-224522','origin':1791585922853839248,'stop':1791585927709765830,'file':'/records/20261009T223912Z-557353c3/samples-000006.jsonl.gz','sha':'796e4bea8072034da39117ded38f6aa3b2b562eeddf5cc7d2a86abe4d0b9f7c2'}
]
out=[]
for case in cases:
 p=pathlib.Path(case['file']);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==case['sha']
 origin=case['origin']; rows=[]; command=None
 with gzip.open(p,'rt') as f:
  for l in f:
   r=json.loads(l);ns=r.get('unix_ns',0);topic=r.get('topic');t=(ns-origin)/1e9
   if t>12: continue
   if topic=='rt/dex3/left/cmd':
    m=decode(r)['message']; command={'time':t,'q':[x['q'] for x in m['motor_cmd']],'kp':[x['kp'] for x in m['motor_cmd']]}
   if not -8<=t<=12 or topic!='rt/lf/dex3/left/state':continue
   m=decode(r)['message'];s=m['press_sensor_state'];mot=m['motor_state']
   rows.append({'t':t,'unix_ns':str(ns),'q':[x['q'] for x in mot],'dq':[x['dq'] for x in mot],'temperatures':[x['temperature'] for x in mot],'faults':[x['motorstate'] for x in mot],'pressures':[x['pressure'] for x in s],'lost':[x['lost'] for x in s],'cmd':command})
 baseline=[r for r in rows if -8<=r['t']<-1];cal=[]
 for i in range(len(rows[0]['pressures'])):
  for j in range(12):
   v=[r['pressures'][i][j] for r in baseline];med=statistics.median(v);mad=statistics.median(abs(x-med) for x in v);cal.append((i,j,med,max(2000.,6.*mad)))
 for r in rows:
  hits=[{'sensor':i,'channel':j,'rise_raw':r['pressures'][i][j]-base} for i,j,base,threshold in cal if r['pressures'][i][j]-base>=threshold]
  r['tactile_hits']=hits;r['tactile_channels']=len(hits);r['selected_pressure_raw']=[r['pressures'][i][j] for i,j in [(1,3),(3,6),(4,11),(5,6)]]
  del r['pressures']
 case['origin']=str(origin);case['stop']=str(case['stop']);case['stop_s']=(int(case['stop'])-origin)/1e9;case['rows']=rows;case['thresholds']=[{'sensor':i,'channel':j,'baseline_median':base,'threshold_raw':threshold} for i,j,base,threshold in cal];out.append(case)
print(json.dumps({'cases':out,'contact_threshold':'rise >= max(2000 raw,6 MAD) per channel; baseline [-8,-1) s','timestamp_basis':'recorder callback receipt'},separators=(',',':')))
