# Подбор сдвига каждой ленты вдоль лозы (ph) и зеркала так, чтобы украшения (не стебель) разных лоз не накладывались.
# Лента = целая петля лозы, поэтому ph свободен в [0,1). Работает на уменьшенной плитке (R=512), перебор по координатному спуску.
#   python -I узлы/подбор_сдвигов.py  → узлы/сдвиги_подобраны.json
import json, math, itertools, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
R=512; S=R/2048
LINES=dict(A=(1,1,.00,0,'по_узлам_A'),B=(1,-1,.40,0,'по_узлам_B'),C=(1,2,.15,0,'по_узлам_C'),D=(2,1,.62,0,'по_узлам_D'),E=(0,1,0,.55,'по_узлам_E'))
def vine_only(im):
    arr=np.array(im); x=8
    ys=np.where(arr[:,x,3]>200)[0]; g=np.split(ys,np.where(np.diff(ys)>3)[0]+1)[-1]; sy=int(g.mean())
    seed=arr[sy-2:sy+3,x-2:x+3,:3].reshape(-1,3).mean(axis=0)
    d=np.abs(arr[...,:3].astype(int)-seed[None,None,:]).max(axis=2)
    lab,_=ndi.label((d<28)&(arr[...,3]>200),structure=np.ones((3,3)))
    m=ndi.binary_dilation(lab==lab[sy,x],iterations=3)&(arr[...,3]>40)
    return m
def piece(a,T,flip):
    # a: массив RGBA-маски (alpha) ленты → 3× плитка, масштаб, вырез одного периода
    H0,W0=a.shape; k=T/W0
    tri=np.tile(a,(1,3)); im=Image.fromarray(tri).resize((int(round(W0*3*k)),int(round(H0*k))),Image.BILINEAR)
    t=int(round(T)); p=im.crop((t,0,2*t,im.height))
    return p.transpose(Image.FLIP_TOP_BOTTOM) if flip else p
def place(pc,k,flip,ph):
    p,q,c,x0,_=LINES[k]; loop=R*math.hypot(p,q); deg=math.degrees(math.atan2(q,p))
    rc=pc.rotate(-deg,resample=Image.BILINEAR,expand=True)
    d=np.array([p,q])/math.hypot(p,q); start=np.array([x0*R,c*R]); cen=(start+d*(0.5+ph)*loop)%R
    out=np.zeros((R,R),np.uint8); w,h=rc.size; ra=np.array(rc)
    x0_,y0_=int(round(cen[0]-w/2)),int(round(cen[1]-h/2))
    for kx in range(-2,3):
        for ky in range(-2,3):
            xx,yy=x0_+kx*R,y0_+ky*R
            if xx>=R or yy>=R or xx+w<=0 or yy+h<=0: continue
            sx0,sy0=max(0,-xx),max(0,-yy); sx1,sy1=min(w,R-xx),min(h,R-yy)
            out[yy+sy0:yy+sy1,xx+sx0:xx+sx1]=np.maximum(out[yy+sy0:yy+sy1,xx+sx0:xx+sx1],ra[sy0:sy1,sx0:sx1])
    return out>60
cache={}
for k,(p,q,c,x0,name) in LINES.items():
    im=Image.open(f'лента/{name}.png').convert('RGBA'); a=np.array(im)[...,3]; v=vine_only(im)
    dec=((a>40)&~v).astype(np.uint8)*255; allp=(a>40).astype(np.uint8)*255
    T=R*math.hypot(p,q); cache[k]={f:(piece(dec,T,f),piece(allp,T,f)) for f in (False,True)}
PHS=np.linspace(0,1,48,endpoint=False)
renders={}
def get(k,f,ph):
    key=(k,f,round(ph,4))
    if key not in renders: renders[key]=(place(cache[k][f][0],k,f,ph),place(cache[k][f][1],k,f,ph))
    return renders[key]
def cost(state):
    tot=0; ks=list(state)
    R_={k:get(k,*state[k]) for k in ks}
    for a,b in itertools.combinations(ks,2):
        da,aa=R_[a]; db,ab=R_[b]
        tot+=(da&db).sum()*1.0+(da&ab).sum()*0.35+(db&aa).sum()*0.35
    return tot
init=json.load(open('узлы/ленты_параметры.json',encoding='utf8'))
state={k:(False,init[k]['ph']) for k in LINES}
best=cost(state); print('старт',best)
for rnd in range(3):
    for k in LINES:
        for f in (False,True):
            for ph in PHS:
                s2=dict(state); s2[k]=(f,float(ph)); c_=cost(s2)
                if c_<best: best,state=c_,s2
    print('круг',rnd,best,{k:(v[0],round(v[1],3)) for k,v in state.items()})
json.dump({k:dict(flip=v[0],ph=v[1]) for k,v in state.items()},open('узлы/сдвиги_подобраны.json','w'),ensure_ascii=False)
