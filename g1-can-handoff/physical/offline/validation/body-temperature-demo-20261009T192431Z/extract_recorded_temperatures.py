import base64,gzip,hashlib,json,sys,time
from pathlib import Path
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
path=Path(sys.argv[1]); expected=sys.argv[2]; start=int(sys.argv[3]); end=int(sys.argv[4]); width=250000000
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest==expected,'Recorder checksum mismatch'
bins={}; count=0; first=None; last=None; prev=None; maxgap=0; peaks=[]; peak=-999; invalid=0; above=0; first_above=None
for line in gzip.open(path,'rt'):
 row=json.loads(line); t=row['unix_ns']
 if row.get('topic')!='rt/lowstate' or not start<=t<=end: continue
 if 'cdr_base64' in row:
  msg=LowState_.deserialize(base64.b64decode(row['cdr_base64'],validate=True)); temps=[list(m.temperature) for m in msg.motor_state[:29]]
 else: temps=[m['temperature'] for m in row['message']['motor_state'][:29]]
 if len(temps)!=29 or any(len(x)!=2 for x in temps): invalid+=1; continue
 count+=1; first=t if first is None else min(first,t); last=t if last is None else max(last,t)
 if prev is not None: maxgap=max(maxgap,t-prev)
 prev=t
 mx=max(max(x) for x in temps)
 if mx>peak: peak=mx; peaks=[{'unix_ns':t,'temperatures':temps,'receive_sequence':row.get('receive_sequence')}]
 if mx>=85:
  above+=1
  if first_above is None: first_above={'unix_ns':t,'temperatures':temps,'receive_sequence':row.get('receive_sequence')}
 k=(t-start)//width
 if k not in bins: bins[k]={'index':k,'first_ns':t,'last_ns':t,'count':0,'min':[x[:] for x in temps],'max':[x[:] for x in temps],'first':temps,'last':temps}
 b=bins[k];b['count']+=1
 if t<b['first_ns']: b['first_ns']=t;b['first']=temps
 if t>b['last_ns']: b['last_ns']=t;b['last']=temps
 for j in range(29):
  for sensor in range(2): b['min'][j][sensor]=min(b['min'][j][sensor],temps[j][sensor]);b['max'][j][sensor]=max(b['max'][j][sensor],temps[j][sensor])
print(json.dumps({'file':str(path),'sha256':digest,'window_start_ns':start,'window_end_ns':end,'bucket_width_ns':width,'records':count,'invalid_temperature_rows':invalid,'first_ns':first,'last_ns':last,'max_gap_ns':maxgap,'peak_c':peak,'peak_records':peaks,'at_or_above_85_records':above,'first_at_or_above_85':first_above,'bins':[bins[k] for k in sorted(bins)]},separators=(',',':')))

