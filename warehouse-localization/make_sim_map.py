"""Surveyed simulation blueprint for AMCL validation, not a real warehouse map."""
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
root=Path(__file__).resolve().parent/'evidence';root.mkdir(exist_ok=True)
res=.05;origin=(-6.5,-5.5);w,h=260,220
yy,xx=np.mgrid[:h,:w];x=origin[0]+(xx+.5)*res;y=origin[1]+(yy+.5)*res
occupied=(abs(x)>=5.92)|(abs(y)>=4.92)
for sx in (-1.7,1.7):occupied|=(abs(x-sx)<=.55)&(abs(y)<=2.)
img=np.where(occupied,0,254).astype('uint8')
Image.fromarray(np.flipud(img)).save(root/'map.pgm')
(root/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=res,origin=[*origin,0.],negate=0,occupied_thresh=.65,free_thresh=.196)))
