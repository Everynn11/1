# Накладывает схему (кривые лоз + круги) из узлы/схема_v3.json на собранную плитку, чтобы сравнить план и результат.
#   python -I узлы/наложение_схемы.py --tile плитка_схема_v3_a.png --out узлы/наложение_v3.png
import argparse, json, math
import numpy as np
from PIL import Image, ImageDraw
ap=argparse.ArgumentParser(); ap.add_argument('--tile',default='плитка_схема_v3_a.png'); ap.add_argument('--schema',default='узлы/схема_v3.json'); ap.add_argument('--out',default='узлы/наложение_v3.png'); a=ap.parse_args()
R=2048; PQ=dict(A=(1,1),B=(1,-1),C=(1,2),D=(2,1)); COL=dict(A=(40,90,220),B=(220,50,50),C=(30,150,60),D=(240,140,10))
st=json.load(open(a.schema,encoding='utf8'))
def curve(k,s):
    prm=st['prm'][k]; p,q=PQ[k]; n=math.hypot(p,q); dx,dy=p/n,q/n
    off=prm['amp']*np.sin(2*np.pi*prm['cyc']*s+prm['ph'])
    return ((prm['x0']+p*s)*R-dy*off)%R,((prm['c']+q*s)*R+dx*off)%R
bg=Image.open(a.tile).convert('RGB'); im=Image.blend(bg,Image.new('RGB',bg.size,(255,255,255)),0.35)
d=ImageDraw.Draw(im)
for k in PQ:
    xs,ys=curve(k,np.arange(3000)/3000)
    for x,y in zip(xs,ys): d.ellipse([x-3,y-3,x+3,y+3],fill=COL[k])
    cx,cy=curve(k,np.array(st['s'][k]))
    for x,y in zip(cx,cy):
        for ox in (-R,0,R):
            for oy in (-R,0,R): d.ellipse([x+ox-150,y+oy-150,x+ox+150,y+oy+150],outline=(0,0,0),width=5)
im.save(a.out); print('готово')
