"""Plot archived NX2 temperatures; never subscribes to or commands hardware."""
import csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROOT=Path(__file__).parent
names=([f"{s}_{j}" for s in ("left","right") for j in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")]
       +["waist_yaw","waist_roll","waist_pitch"]
       +[f"{s}_{j}" for s in ("left","right") for j in ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw")])
segments=[json.loads((ROOT/f"segment-{i}.json").read_text()) for i in ("006","007","008")]
events=json.loads((ROOT/"events.json").read_text())["events"]
start=next(e["unix_ns"] for e in events if e["phase"]=="demo_start_accepted")
failure=next(e for e in events if e["phase"]=="primary_failure")
stop=failure["unix_ns"]
width=segments[0]["bucket_width_ns"]; window=segments[0]["window_start_ns"]
bins={}
for segment in segments:
 for b in segment["bins"]:
  k=b["index"]
  if k not in bins: bins[k]=b;continue
  a=bins[k]; a["count"]+=b["count"]
  a["min"]=np.minimum(a["min"],b["min"]).tolist();a["max"]=np.maximum(a["max"],b["max"]).tolist()
  if b["first_ns"]<a["first_ns"]: a["first_ns"]=b["first_ns"];a["first"]=b["first"]
  if b["last_ns"]>a["last_ns"]: a["last_ns"]=b["last_ns"];a["last"]=b["last"]
keys=np.arange(min(bins),max(bins)+1)
time=(window+keys*width-start)/1e9
high=np.full((len(keys),29,2),np.nan); low=high.copy(); counts=np.zeros(len(keys),dtype=int)
for i,k in enumerate(keys):
 if int(k) in bins:
  high[i]=bins[int(k)]["max"];low[i]=bins[int(k)]["min"];counts[i]=bins[int(k)]["count"]
maxima=high.max(axis=2); minima=low.max(axis=2)
demo=(time>=0)&(time<=(stop-start)/1e9)
ranking=np.argsort(np.nanmax(maxima[demo],axis=0))[::-1]
first_above=min((s["first_at_or_above_85"] for s in segments if s["first_at_or_above_85"]),key=lambda r:r["unix_ns"])
first_temps=np.asarray(first_above["temperatures"]);trigger_indices=np.flatnonzero(first_temps.max(axis=1)>=85).tolist()
closest_start=min(bins.values(),key=lambda b:abs(b["first_ns"]-start))
before_stop=[b for b in bins.values() if b["first_ns"]<=stop]
near_stop=max(before_stop,key=lambda b:b["first_ns"])
summary={
 "device":"Unitree G1 Nx 2","recorder_session":"20261009T191749Z-f6bb3fab",
 "start_utc":datetime.fromtimestamp(start/1e9,timezone.utc).isoformat(),
 "error_utc":datetime.fromtimestamp(stop/1e9,timezone.utc).isoformat(),
 "error":failure["error"],"error_source":failure["source"],"error_after_start_s":(stop-start)/1e9,
 "records":sum(s["records"] for s in segments),"invalid_temperature_rows":sum(s["invalid_temperature_rows"] for s in segments),
 "empty_250ms_bins":int((counts==0).sum()),
 "first_threshold_crossing_s":(first_above["unix_ns"]-start)/1e9,
 "first_threshold_joints":[{"index":j,"name":names[j],"sensors_c":first_temps[j].tolist()} for j in trigger_indices],
 "start_hottest_c":int(np.max(closest_start["first"])),
 "ranking_at_or_before_stop":[{"index":int(j),"name":names[j],"peak_c":int(np.nanmax(maxima[demo,j])),"near_stop_sensors_c":near_stop["last"][j]} for j in ranking],
 "max_record_gap_ms":max(s["max_gap_ns"] for s in segments)/1e6,
 "segments":[{k:s[k] for k in ["file","sha256","records","first_ns","last_ns","max_gap_ns","peak_c","at_or_above_85_records"]} for s in segments],
 "display":"250 ms peak bins, each joint uses hotter of two reported sensors; times are recorder callback receipt times.",
 "coverage":"All delivered rt/lowstate rows in the selected sealed interval; upstream DDS loss is unknown."
}
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
with (ROOT/"body-temperatures.csv").open("w",newline="") as out:
 writer=csv.writer(out);writer.writerow(["seconds_since_start","body_samples"]+[f"{n}_sensor{s}_{v}_c" for n in names for s in (0,1) for v in ("min","max")])
 for i,t in enumerate(time): writer.writerow([f"{t:.3f}",int(counts[i])]+[int(arr[i,j,s]) if np.isfinite(arr[i,j,s]) else "" for j in range(29) for s in (0,1) for arr in (low,high)])
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig=plt.figure(figsize=(14,11))
grid=fig.add_gridspec(2,2,height_ratios=[1,1.65],width_ratios=[1,.025],hspace=.16,wspace=.035)
ax=fig.add_subplot(grid[0,0]);heat=fig.add_subplot(grid[1,0],sharex=ax);color_axis=fig.add_subplot(grid[1,1])
colors=["#d64541","#31688e","#35a88b","#7657a6","#c19a38"]
for c,j in zip(colors,ranking[:5]):
 ax.step(time,maxima[:,j],where="post",color=c,label=names[j].replace("_"," "),linewidth=1.7)
 ax.fill_between(time,minima[:,j],maxima[:,j],step="post",color=c,alpha=.08)
ax.axhline(85,color="#b2252d",ls="--",lw=1.2,label="85°C stop guard")
stop_s=(stop-start)/1e9
ax.axvline(stop_s,color="#b2252d",lw=1.4)
ax.annotate(f"Guard stops demo\n{stop_s:.2f} s after Start",xy=(stop_s,85),xytext=(stop_s-34,92),fontsize=10,color="#b2252d",arrowprops={"arrowstyle":"->","color":"#b2252d"})
ax.set_ylabel("Reported temperature (°C)")
ax.set_ylim(25,max(98,float(np.nanmax(maxima))+6))
ax.grid(alpha=.2);ax.legend(loc="upper left",ncol=3,fontsize=9)
phases=[("Planning / owned hold",next(e["unix_ns"] for e in events if e["phase"]=="control_settled"),next(e["unix_ns"] for e in events if e["phase"]=="staging_segment")),
        ("Staging",next(e["unix_ns"] for e in events if e["phase"]=="staging_segment"),next(e["unix_ns"] for e in events if e["phase"]=="approach")),
        ("Approach",next(e["unix_ns"] for e in events if e["phase"]=="approach"),next(e["unix_ns"] for e in events if e["phase"]=="align")),
        ("Align",next(e["unix_ns"] for e in events if e["phase"]=="align"),next(e["unix_ns"] for e in events if e["phase"]=="reach")),
        ("Reach",next(e["unix_ns"] for e in events if e["phase"]=="reach"),next(e["unix_ns"] for e in events if e["phase"]=="close")),
        ("Stopped / holding",stop,next(e["unix_ns"] for e in events if e["phase"]=="released"))]
for idx,(label,a,b) in enumerate(phases):
 left=(a-start)/1e9;right=(b-start)/1e9
 ax.axvspan(left,right,color="#41617e",alpha=.045 if idx%2 else .1)
 if right-left>4: ax.text((left+right)/2,27,label,ha="center",va="bottom",fontsize=9,rotation=0)
heat.imshow(maxima.T,aspect="auto",origin="upper",extent=[time[0],time[-1]+width/1e9,28.5,-.5],cmap="inferno",norm=Normalize(30,90),interpolation="nearest")
heat.set_yticks(range(29),[n.replace("_"," ") for n in names],fontsize=8)
heat.axvline(stop_s,color="#39d9ee",lw=1.5)
for boundary in (5.5,11.5,14.5,21.5): heat.axhline(boundary,color="white",lw=.7,alpha=.6)
heat.set_xlabel("Seconds since Start demo was accepted")
heat.set_ylabel("All 29 body joints")
fig.colorbar(heat.images[0],cax=color_axis,label="Hotter sensor in each 250 ms bin (°C)")
ax.set_title("G1 NX2 · Body temperatures through the failed Coke demo",loc="left",fontsize=17,pad=35)
ax.text(0,1.035,f"2026-10-09 · Start {summary['start_utc'][11:19]} UTC · Recorder data, not simulation",transform=ax.transAxes,color="#555",fontsize=10)
fig.text(.15,.025,f"{summary['records']:,} body samples · 250 ms maxima preserve spikes · Two sensors per motor · Callback receipt-time alignment",fontsize=9,color="#555")
fig.subplots_adjust(left=.17,right=.94,top=.9,bottom=.08)
fig.savefig(ROOT/"body-temperature-demo.png",dpi=150,facecolor="white")
fig.savefig(ROOT/"body-temperature-demo.pdf",facecolor="white")
plt.close(fig)
print(json.dumps({k:summary[k] for k in ["records","error_after_start_s","first_threshold_crossing_s","first_threshold_joints","start_hottest_c","empty_250ms_bins"]},indent=2))
print(json.dumps(summary["ranking_at_or_before_stop"][:5],indent=2))
