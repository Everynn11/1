# Подбор сдвига ленты вдоль лозы (ph), зеркала и параллельного смещения лозы (c / x0) по РЕАЛЬНЫМ лентам:
# минимизируем наложение украшений на украшения и на стебли других лоз (размер 512 px). → узлы/подбор_v2.json
import json, math, itertools
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
R=512
L0=dict(A=(1,1,.00,0),B=(1,-1,.40,0),C=(1,2,.15,0),D=(2,1,.62,0),E=(0,1,0,.55))
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
    out=np.zeros((R,R),bool); w,h=rc.size; ra=np.array(rc)>60; X0,Y0=int(round(cen[0]-w/2)),int(round(cen[1]-h/2))
    for kx in range(-2,3):
        for ky in range(-2,3):
            xx,yy=X0+kx*R,Y0+ky*R
            if xx>=R or yy>=R or xx+w<=0 or yy+h<=0: continue
            sx0,sy0=max(0,-xx),max(0,-yy); sx1,sy1=min(w,R-xx),min(h,R-yy)
            out[yy+sy0:yy+sy1,xx+sx0:xx+sx1]|=ra[sy0:sy1,sx0:sx1]
    return out
P={}
for k,(p,q,c,x0) in L0.items():
    im=Image.open(f'лента/по_узлам_{k}.png').convert('RGBA'); a=np.array(im)[...,3]; v=vine_only(im); T=R*math.hypot(p,q)
    P[k]={f:(piece(((a>40)&~v).astype(np.uint8)*255,T,f),piece((a>40).astype(np.uint8)*255,T,f)) for f in (False,True)}
cr={}
def get(k,f,ph,c,x0):
    key=(k,f,round(ph,4),round(c,4),round(x0,4))
    if key not in cr:
        if len(cr)>8000: cr.clear()
        p,q=L0[k][:2]; cr[key]=(place(P[k][f][0],p,q,c,x0,ph),place(P[k][f][1],p,q,c,x0,ph))
    return cr[key]
def pair(a,b): return (a[0]&b[0]).sum()+(a[0]&b[1]).sum()+(b[0]&a[1]).sum()
def pcost(st,k):
    r=get(k,*st[k]); return sum(pair(r,get(j,*st[j])) for j in st if j!=k)
def total(st): return sum(pair(get(a,*st[a]),get(b,*st[b])) for a,b in itertools.combinations(st,2))
init=json.load(open('узлы/сдвиги_подобраны.json',encoding='utf8'))
st={k:(init[k]['flip'],init[k]['ph'],L0[k][2],L0[k][3]) for k in L0}
print('старт',total(st))
PHS=np.linspace(0,1,36,endpoint=False); CS=np.linspace(0,1,12,endpoint=False)
for rnd in range(3):
    for k in L0:
        cur=pcost(st,k); vert=L0[k][0]==0
        for f in (False,True):
            for ph in PHS:
                for c in ([L0[k][2]] if k=='A' else CS):
                    cc,xx=((L0[k][2],c) if vert else (c,L0[k][3]))
                    cand=dict(st); cand[k]=(f,float(ph),float(cc),float(xx)); v=pcost(cand,k)
                    if v<cur: cur,st=v,cand
    print('круг',rnd,total(st),{k:(v[0],round(v[1],3),round(v[2],2),round(v[3],2)) for k,v in st.items()},flush=True)
json.dump({k:dict(flip=v[0],ph=v[1],c=v[2],x0=v[3]) for k,v in st.items()},open('узлы/подбор_v2.json','w'),ensure_ascii=False)
