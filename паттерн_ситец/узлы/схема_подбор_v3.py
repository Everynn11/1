# СХЕМА плитки с нуля: 5 изогнутых лоз (замкнутых на торе) и круги-украшения.
# Подбираем изгиб (амплитуда/фаза/число волн), смещение лоз и положения кругов, чтобы круги были как можно дальше
# друг от друга и в каждом круге оказывалось не больше одной чужой лозы. Пересечения вне кругов — голые стебли.
#   python -I узлы/схема_подбор.py --seed 1 --iters 40000  → узлы/схема_v2.json, узлы/схема_v2.png
import argparse, json, math, random
import numpy as np
from PIL import Image, ImageDraw
ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--iters',type=int,default=40000)
ap.add_argument('--r',type=float,default=150); ap.add_argument('--out',default='узлы/схема_v2'); a=ap.parse_args()
R=2048; random.seed(a.seed); np.random.seed(a.seed)
NAMES='ABCD'; PQ=dict(A=(1,1),B=(1,-1),C=(1,2),D=(2,1))
COUNT=dict(A=3,B=3,C=4,D=4); COL=dict(A=(40,90,220),B=(220,50,50),C=(30,150,60),D=(240,140,10))
N=500
def curve(k,prm,s):
    p,q=PQ[k]; n=math.hypot(p,q); dx,dy=p/n,q/n; loop=R*n
    off=prm['amp']*np.sin(2*np.pi*prm['cyc']*s+prm['ph'])
    return ((prm['x0']+p*s)*R-dy*off)%R,((prm['c']+q*s)*R+dx*off)%R
def tdist(x1,y1,x2,y2):
    dx=np.abs(x1-x2); dy=np.abs(y1-y2); return np.hypot(np.minimum(dx,R-dx),np.minimum(dy,R-dy))
S=np.arange(N)/N
def evaluate(st):
    P=st['prm']; pts={k:curve(k,P[k],S) for k in NAMES}
    ctr=[]
    for k in NAMES:
        xs,ys=curve(k,P[k],np.array(st['s'][k])); ctr+=[(k,x,y) for x,y in zip(xs,ys)]
    mind=1e9
    for i in range(len(ctr)):
        for j in range(i+1,len(ctr)):
            mind=min(mind,tdist(ctr[i][1],ctr[i][2],ctr[j][1],ctr[j][2]))
    pen=0
    for k,x,y in ctr:
        fl=0
        for k2 in NAMES:
            if k2==k: continue
            if tdist(x,y,pts[k2][0],pts[k2][1]).min()<a.r*0.85: fl+=1
        pen+=max(0,fl-1)
    gx,gy=np.meshgrid(np.arange(0,R,64),np.arange(0,R,64)); gx=gx.ravel(); gy=gy.ravel()
    near=np.full(gx.shape,1e9)
    for k,x,y in ctr: near=np.minimum(near,tdist(gx,gy,x,y))
    cov=near.max()
    return -cov-3*max(0,440-mind)-40*pen,mind,pen,cov
def rnd():
    P={k:dict(amp=random.uniform(60,200),cyc=random.choice([1,2,2,3]),ph=random.uniform(0,6.28),
              c=random.random(),x0=0.0) for k in NAMES}
    P['A']['c']=0.0
    return dict(prm=P,s={k:sorted(random.random() for _ in range(COUNT[k])) for k in NAMES})
best=None
for restart in range(4):
    st=rnd(); sc=evaluate(st)[0]
    for it in range(a.iters//4):
        c=json.loads(json.dumps(st)); k=random.choice(NAMES); r=random.random()
        if r<.45: c['s'][k]=sorted((x+random.gauss(0,.03))%1 for x in c['s'][k])
        elif r<.6: c['prm'][k]['amp']=min(220,max(50,c['prm'][k]['amp']+random.gauss(0,20)))
        elif r<.75: c['prm'][k]['ph']=(c['prm'][k]['ph']+random.gauss(0,.5))%6.283
        elif r<.9 and k!='A':
            c['prm'][k]['c']=(c['prm'][k]['c']+random.gauss(0,.04))%1
        else: c['prm'][k]['cyc']=random.choice([1,2,3])
        v=evaluate(c)[0]
        if v>=sc: st,sc=c,v
    print('перезапуск',restart,evaluate(st),flush=True)
    if best is None or sc>best[0]: best=(sc,st)
sc,st=best; print('лучшее',evaluate(st))
json.dump(st,open(a.out+'.json','w'),ensure_ascii=False)
SS=2; W=R*SS; im=Image.new('RGB',(W,W),(255,255,255)); d=ImageDraw.Draw(im)
for k in NAMES:
    xs,ys=curve(k,st['prm'][k],np.arange(4000)/4000)
    for x,y in zip(xs,ys):
        for ox in (-R,0,R):
            for oy in (-R,0,R):
                X,Y=(x+ox)*SS,(y+oy)*SS
                if -20<=X<=W+20 and -20<=Y<=W+20: d.ellipse([X-7,Y-7,X+7,Y+7],fill=COL[k])
    cx,cy=curve(k,st['prm'][k],np.array(st['s'][k]))
    for x,y in zip(cx,cy):
        for ox in (-R,0,R):
            for oy in (-R,0,R):
                X,Y=(x+ox)*SS,(y+oy)*SS; r=a.r*SS
                d.ellipse([X-r,Y-r,X+r,Y+r],outline=COL[k],width=8)
im.resize((R,R),Image.LANCZOS).save(a.out+'.png')
