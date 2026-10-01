"""Render measured trajectories, preserving movement speed between cuts."""
import argparse,json,subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from .whole_body import make
from . import bar_film as bar

OUTPUT=bar.PROJECT/'results'/'generalization-physics'

def short_segments(a):
    times=a['time'];phases=a['phase']
    selected=[]
    for mask in [times<2, np.isin(phases,['reach','close','lift','offer']), np.isin(phases,['carry_step_1','carry_step_0','hold_after_steps'])]:
        indices=np.flatnonzero(mask)
        if len(indices):selected.extend(indices.tolist())
    if not np.any(np.isin(phases,['close','lift'])):
        selected.extend(np.flatnonzero(times>=max(2,times[-1]-6)).tolist())
    return list(dict.fromkeys(selected))

def render(cases):
    width,height,fps=960,540,10
    target=OUTPUT/'generalization-evidence.mp4'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',20)
    small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16)
    rendered=[]
    try:
        for name in cases:
            folder=OUTPUT/name;result=json.loads((folder/'result.json').read_text())
            scenario=result.get('scenario',result);p=make(scenario['height'],scenario['x'],scenario['y'],scenario.get('delivery_height',.75))
            archive=np.load(folder/'states.npz');a={k:archive[k] for k in archive.files}
            if result.get('kind')=='full_route':
                # Stored at 20 Hz: use consecutive 10 Hz samples within each cut.
                indices=[]
                for phase in dict.fromkeys(a['phase']):
                    segment=np.flatnonzero(a['phase']==phase)[::2]
                    if phase.startswith(('walk','round','approach_table')):
                        if not result['success'] and phase==a['phase'][-1]:segment=segment[-70:]
                        else:
                            center=len(segment)//2;segment=segment[max(0,center-25):center+25]
                    elif phase=='prepare_reach':segment=segment[:20]
                    indices.extend(segment.tolist())
            else:indices=short_segments(a)
            renderer=mujoco.Renderer(p.model,height=height,width=width)
            options=mujoco.MjvOption();options.geomgroup[:]=1
            first=True
            try:
                for count,index in enumerate(indices):
                    p.data.qpos[:]=a['qpos'][index];mujoco.mj_forward(p.model,p.data)
                    phase=str(a['phase'][index])
                    if phase.startswith(('carry_step','hold_after','stand','arms_at','prepare','walk_behind')):camera='aisle_walk'
                    elif phase in ('round_counter_end','approach_table_corner','walk_along_table','adjust_table_stance'):camera='route_wide'
                    elif phase in ('restore_upright','align_over_table','lower_can','release_on_table','withdraw_hand'):camera='table_place'
                    elif phase=='finished':camera='table_finish'
                    else:camera='bar_close'
                    renderer.update_scene(p.data,camera=camera,scene_option=options)
                    image=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,width,66),fill=(17,22,20))
                    draw.text((18,10),f"{name}   Counter {scenario['height']*100:.0f} cm   Can x {scenario['x']*100:g}, y {scenario['y']*100:g} cm",font=font,fill=(241,245,243))
                    status='PASS' if result['success'] else 'FAIL'
                    scope='Complete delivery' if result.get('kind')=='full_route' else 'Pickup + two physical steps'
                    draw.text((18,38),f'{scope}  |  {status}  |  Simulation time {a["time"][index]:.1f} s',font=small,fill=(241,245,243))
                    draw.rectangle((0,height-36,width,height),fill=(17,22,20))
                    draw.text((18,height-28),'MUJOCO CONTACT PHYSICS  |  Original movement speed; cuts between phases',font=small,fill=(241,245,243))
                    encoder.stdin.write(image.tobytes())
                    if first:image.save(folder/'evidence-preview.jpg');first=False
                if not result['success']:
                    draw.rectangle((0,height-85,width,height),fill=(17,22,20))
                    draw.text((18,height-70),result.get('failure','Failed')[:110],font=small,fill=(241,245,243))
                    for _ in range(25):encoder.stdin.write(image.tobytes())
            finally:renderer.close()
            rendered.append(dict(id=name,frames=len(indices),success=result['success']))
            print('RENDERED',name,len(indices),flush=True)
    finally:
        encoder.stdin.close();encoder.wait()
    if encoder.returncode:raise RuntimeError('Video encoding failed')
    (OUTPUT/'video-manifest.json').write_text(json.dumps(dict(cases=rendered,fps=fps,source='Recorded physics states; no generated motion'),indent=2))
    print(target,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+');args=parser.parse_args();render(args.cases)
