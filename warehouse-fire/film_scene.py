"""Presentation-only warehouse detail, without changing sensor calibration."""
import xml.etree.ElementTree as ET
import mujoco
from twin import RACKS

def dress(twin,path):
    root=ET.parse(path).getroot(); world=root.find('worldbody')
    ET.SubElement(root,'compiler',angle='radian')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='concrete',type='2d',builtin='checker',rgb1='.30 .32 .32',rgb2='.32 .34 .34',width='512',height='512',random='.04',mark='random',markrgb='.25 .26 .26')
    ET.SubElement(asset,'material',name='concrete',texture='concrete',texrepeat='40 40',texuniform='true',reflectance='.1',specular='.15',shininess='.05')
    for name,color,spec in [('steel','.18 .25 .29 1','.6'),('rackblue','.06 .20 .31 1','.4'),('cardboard','.54 .37 .20 1','.05'),('wood','.33 .23 .12 1','.1'),('pipe','.5 .52 .53 1','.75'),('white','.86 .85 .80 1','.1'),('yellow','.86 .55 .08 1','.1')]:
        ET.SubElement(asset,'material',name=name,rgba=color,specular=spec,shininess='.3')
    ET.SubElement(asset,'material',name='fire',rgba='1 .5 .03 1',emission='1')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
    ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
    ET.SubElement(visual,'headlight',ambient='.20 .20 .23',diffuse='.45 .45 .45',specular='.15 .15 .15')
    ET.SubElement(visual,'map',znear='.001',zfar='100',fogstart='70',fogend='120')
    ET.SubElement(visual,'rgba',haze='.12 .16 .20 1')
    floor=world.find("geom[@name='floor']");floor.set('material','concrete');floor.attrib.pop('rgba',None)
    for g in world.findall('geom'):
        name=g.get('name','')
        if name.startswith('storage_'):g.set('rgba','0 0 0 0')
        if name.startswith(('post_','rack_')):g.set('material','rackblue');g.attrib.pop('rgba',None)
        if name.startswith('zone_'):g.set('rgba','0 0 0 0')
        if name.startswith('housing_'):g.set('rgba','.15 .17 .17 1')
    for body in world.findall('body'):
        if body.get('name','').startswith(('flame_','marker_','drop_')):
            body.find('geom').set('rgba','0 0 0 0')
    for light in world.findall('light'):world.remove(light)
    for x in (8,20,32):
        ET.SubElement(world,'light',pos=f'{x} 15 12',dir='-.15 .1 -1',directional='true',diffuse='.30 .30 .28',specular='.45 .45 .45',castshadow='true' if x==20 else 'false')
    count=0
    def g(kind,pos,size,material='steel',**kwargs):
        nonlocal count
        count+=1
        return ET.SubElement(world,'geom',name=f'detail_{count}',type=kind,pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),material=material,group='3',contype='0',conaffinity='0',**kwargs)
    # Painted aisle lanes and hatching are actual geometry, so perspective and shadows match.
    for x in (5,15,25,35):
        for side in (-1,1):g('box',(x+side*1.8,20,.007),(.035,18,.005),'yellow')
    for x in (2,38):
        for y in range(2,39,2):g('box',(x,y,.007),(.55,.045,.005),'yellow',euler='0 0 -.65')
    for x,y in RACKS:
        for z in (.35,1.55):
            for yy in (-1.45,-.5,.5,1.45):
                for xx in (-.4,.4):
                    g('box',(x+xx,y+yy,z+.3),(.36,.42,.29),'cardboard')
                    # Shipping label and packing tape.
                    g('box',(x+xx,y+yy-.425,z+.35),(.12,.003,.065),'white')
                    g('box',(x+xx,y+yy,z+.593),(.026,.42,.003),'white')
        # Slatted pallet and diagonal rack braces.
        for yy in (-1.7,-1.3,-.9,-.5,-.1,.3,.7,1.1,1.5):g('box',(x,y+yy,.23),(.8,.12,.045),'wood')
        for sx in (-1,1):
            g('capsule',(x+sx*.92,y,1.3),(.026,2.4),'steel',euler='.95 0 0')
    # Tall industrial side walls and corrugated loading doors. South is an intentional cutaway.
    for x in (0,40):g('box',(x,20,4.8),(.12,20,2.8),'white')
    g('box',(20,40,4.8),(20,.12,2.8),'white')
    for x in (5,15,25,35):
        g('box',(x,39.8,2.2),(2.3,.15,2.2),'steel')
        for z in range(1,23):g('box',(x,39.63,z*.19),(2.3,.045,.012),'pipe')
        for side in (-1,1):g('box',(x+side*2.6,39.5,.5),(.15,.15,.5),'yellow')
    # Roof structure, lighting fixtures, sprinkler branch lines.
    for y in (4,12,20,28,36):
        g('box',(20,y,8),(20,.09,.13),'steel')
        for x in range(0,40,4):
            g('capsule',(x+2,y,8.45),(.045,2.05),'steel',euler='0 1.35 0')
        g('cylinder',(20,y,6.55),(.055,19.5),'pipe',euler='0 1.5708 0')
        for x in (5,15,25,35):g('box',(x,y,7.65),(1.5,.18,.05),'white')
    for x in (2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5):
        g('cylinder',(x,20,6.5),(.025,19.5),'pipe',euler='1.5708 0 0')
    for cam in twin.cameras:
        x,y,z=cam['position']
        g('cylinder',(x,y,z-.10),(.055,.05),'steel')
        g('sphere',(x,y,z-.165),(.04,),'pipe')
        g('box',(x,y,z+.2),(.022,.022,.2),'steel')
    # Pallet stacks and packing station beside the loading bay.
    for x in (3,37):
        for y in (6,12,18):
            for z in range(4):g('box',(x,y,.12+z*.2),(.7,.55,.09),'wood')
    for x in (16,24):
        g('box',(x,36,.95),(2,.7,.08),'steel')
        for sx in (-1,1):
            for sy in (-1,1):g('box',(x+sx*1.8,36+sy*.55,.48),(.05,.05,.45),'steel')
        for sx in (-1,0,1):g('box',(x+sx,36,1.35),(.35,.35,.33),'cardboard')
    film_path=path.with_name('warehouse-film.xml');ET.ElementTree(root).write(film_path)
    twin.model=mujoco.MjModel.from_xml_path(str(film_path));twin.data=mujoco.MjData(twin.model)
    mujoco.mj_forward(twin.model,twin.data)
    return film_path
