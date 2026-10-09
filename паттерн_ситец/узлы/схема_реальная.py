# Схема плитки по РЕАЛЬНЫМ лентам: стебли каждой лозы (с настоящими изгибами) тёмным цветом линии, украшения — полупрозрачным.
#   python -I узлы/схема_реальная.py --json узлы/сдвиги_подобраны.json --out узлы/схема_реальная.png
import json, math, argparse, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ap=argparse.ArgumentParser(); ap.add_argument('--json',default='узлы/сдвиги_подобраны.json'); ap.add_argument('--out',default='узлы/схема_реальная.png'); ap.add_argument('--size',type=int,default=1024)
a=ap.parse_args(); R=a.size
LINES=dict(A=(1,1,.00,0),B=(1,-1,.40,0),C=(1,2,.15,0),D=(2,1,.62,0),E=(0,1,0,.55))
RIB=dict(A='по_узлам_A',B='по_узлам_B',C='по_узлам_C',D='по_узлам_D',E='по_узлам_E')
COL=dict(A=(40,90,220),B=(220,50,50),C=(30,150,60),D=(240,140,10),E=(150,50,200))
st=json.load(open(a.json,encoding='utf8'))
def vine_only(im):
    arr=np.array(im); x=8
    ys=np.where(arr[:,x,3]>200)[0]; g=np.split(ys,np.where(np.diff(ys)>3)[0]+1)[-1]; sy=int(g.mean())
    seed=arr[sy-2:sy+3,x-2:x+3,:3].reshape(-1,3).mean(axis=0)
    d=np.abs(arr[...,:3].astype(int)-seed[None,None,:]).max(axis=2)
    lab,_=ndi.label((d<28)&(arr[...,3]>200),structure=np.ones((3,3)))
    return ndi.binary_dilation(lab==lab[sy,x],iterations=3)&(arr[...,3]>40)
def piece(m,T,flip):
    H0,W0=m.shape; k=T/W0
    im=Image.fromarray(np.tile(m,(1,3))).resize((int(round(W0*3*k)),int(round(H0*k))),Image.BILINEAR)
    t=int(round(T)); p=im.crop((t,0,2*t,im.height)); return p.transpose(Image.FLIP_TOP_BOTTOM) if flip else p
def place(pc,p,q,c,x0,ph):
    loop=R*math.hypot(p,q); deg=math.degrees(math.atan2(q,p)); rc=pc.rotate(-deg,resample=Image.BILINEAR,expand=True)
    d=np.array([p,q])/math.hypot(p,q); cen=(np.array([x0*R,c*R])+d*(0.5+ph)*loop)%R
    out=np.zeros((R,R),np.uint8); w,h=rc.size; ra=np.array(rc); X0,Y0=int(round(cen[0]-w/2)),int(round(cen[1]-h/2))
    for kx in range(-2,3):
        for ky in range(-2,3):
            xx,yy=X0+kx*R,Y0+ky*R
            if xx>=R or yy>=R or xx+w<=0 or yy+h<=0: continue
            sx0,sy0=max(0,-xx),max(0,-yy); sx1,sy1=min(w,R-xx),min(h,R-yy)
            out[yy+sy0:yy+sy1,xx+sx0:xx+sx1]=np.maximum(out[yy+sy0:yy+sy1,xx+sx0:xx+sx1],ra[sy0:sy1,sx0:sx1])
    return out>60
img=np.full((R,R,3),255.,float)
for k,(p,q,c0,x00) in LINES.items():
    s=st.get(k,{}); f=s.get('flip',False); ph=s.get('ph',0); c=s.get('c',c0); x0=s.get('x0',x00); rib=s.get('rib',RIB[k])
    im=Image.open(f'лента/{rib}.png').convert('RGBA'); al=np.array(im)[...,3]; v=vine_only(im)
    T=R*math.hypot(p,q)
    vine=place(piece(((al>40)&v).astype(np.uint8)*255,T,f),p,q,c,x0,ph)
    dec=place(piece(((al>40)&~v).astype(np.uint8)*255,T,f),p,q,c,x0,ph)
    col=np.array(COL[k],float)
    img[dec]=img[dec]*0.45+col*0.55
    img[vine]=col*0.8
Image.fromarray(img.astype(np.uint8)).save(a.out)
t=Image.new('RGB',(R*2,R*2))
for x in (0,1):
    for y in (0,1): t.paste(Image.open(a.out),(x*R,y*R))
t.resize((1400,1400),Image.LANCZOS).save(a.out.replace('.png','_2x2.jpg'),quality=88)
print('готово',a.out)
