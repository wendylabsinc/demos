"""Scene props and the ten detector labels. Search never reads their positions."""
from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET
import numpy as np

@dataclass(frozen=True)
class Item:
    name: str
    position: tuple

ITEMS = (
    Item('ball', (-4.2,-3.2)), Item('bottle',(-2.5,-3.6)),
    Item('cup',(0,-3.7)), Item('bowl',(2.4,-3.6)),
    Item('banana',(4.2,-3.2)), Item('apple',(4.4,0)),
    Item('orange',(4.2,3.2)), Item('chair',(1,3.6)),
    Item('suitcase',(-2,3.6)), Item('backpack',(-4.3,1)),
)
NAMES = tuple(item.name for item in ITEMS)

def requested_item(query):
    words = re.findall(r'[a-z]+', query.lower())
    names = set(words) & set(NAMES)
    if len(names) != 1:
        raise ValueError('Ask for exactly one item: ' + ', '.join(NAMES))
    return names.pop()

def add_items(world, positions=None):
    for item in ITEMS:
        x,y = (positions or {}).get(item.name,item.position)
        body=ET.SubElement(world,'body',name='item_'+item.name,pos=f'{x} {y} 0')
        def g(name,typ,pos,size,color,**kw):
            return ET.SubElement(body,'geom',name=item.name+'_'+name,type=typ,
                pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba=color,group='1',**kw)
        if item.name=='ball':
            g('body','sphere',(0,0,.32),(.32,),'1 1 1 1',material='ball_mat')
        elif item.name=='bottle':
            g('body','cylinder',(0,0,.34),(.16,.32),'.10 .60 .80 1')
            g('shoulder','ellipsoid',(0,0,.66),(.16,.16,.12),'.10 .60 .80 1')
            g('neck','cylinder',(0,0,.79),(.075,.11),'.10 .60 .80 1')
            g('cap','cylinder',(0,0,.91),(.085,.03),'.90 .93 .97 1')
            g('label','cylinder',(0,0,.37),(.163,.12),'.90 .93 .97 1')
        elif item.name in ('cup','bowl'):
            # Open vessel assembled from a base and a ring of side panels.
            cup=item.name=='cup'; radius=.22 if cup else .35; height=.46 if cup else .26
            color='.92 .85 .70 1' if cup else '.25 .68 .87 1'
            g('base','cylinder',(0,0,.035),(radius*.85,.035),color)
            for i,angle in enumerate(np.linspace(0,2*np.pi,32,endpoint=False)):
                r=radius*.94
                g(str(i),'box',(r*np.cos(angle),r*np.sin(angle),height/2),(.028,.05,height/2),color,euler=f'0 0 {angle}')
            if cup:
                for i,a in enumerate(np.linspace(-np.pi/2,np.pi/2,13)):
                    g('handle'+str(i),'sphere',(.23+.15*np.cos(a),0,.24+.16*np.sin(a)),(.045,),color)
        elif item.name=='banana':
            for i,a in enumerate(np.linspace(-1.2,1.2,12)):
                g(str(i),'ellipsoid',(.45*np.sin(a),.22*(1-np.cos(a)),.13),(.11,.10,.10),'1 .83 .08 1',euler=f'0 0 {a}')
            g('tip','sphere',(-.43,.13,.13),(.055,),'.26 .20 .07 1')
        elif item.name in ('apple','orange'):
            apple=item.name=='apple'
            g('fruit','ellipsoid',(0,0,.26),(.27,.27,.26),'.85 .08 .06 1' if apple else '1 .42 .025 1')
            if apple:
                g('stem','cylinder',(0,0,.54),(.026,.07),'.26 .15 .06 1')
                g('leaf','ellipsoid',(.08,0,.55),(.12,.055,.018),'.15 .48 .12 1',euler='0 -.3 0')
        elif item.name=='chair':
            g('seat','box',(0,0,.49),(.36,.34,.055),'.75 .19 .13 1')
            g('back','box',(0,.29,.87),(.36,.055,.34),'.75 .19 .13 1')
            for i,(px,py) in enumerate([(-.28,-.26),(.28,-.26),(-.28,.26),(.28,.26)]):
                g('leg'+str(i),'box',(px,py,.24),(.035,.035,.24),'.15 .17 .20 1')
        elif item.name=='suitcase':
            g('case','box',(0,0,.50),(.32,.20,.43),'.30 .24 .62 1')
            for i,x in enumerate([-.24,.24]):
                g('wheel'+str(i),'sphere',(x,0,.065),(.065,),'.07 .08 .09 1')
                g('handleleg'+str(i),'cylinder',(x*.5,0,1.04),(.022,.13),'.65 .68 .70 1')
            g('handle','box',(0,0,1.18),(.15,.035,.025),'.07 .08 .09 1')
        else:
            g('bag','ellipsoid',(0,0,.40),(.30,.21,.40),'.13 .41 .25 1')
            g('pocket','ellipsoid',(0,-.20,.29),(.24,.09,.21),'.20 .55 .34 1')
            g('loop','capsule',(0,0,.82),(.026,.10),'.06 .16 .10 1',euler='0 1.5708 0')
            for i,x in enumerate([-.18,.18]):
                g('strap'+str(i),'capsule',(x,.20,.4),(.04,.27),'.06 .16 .10 1')
