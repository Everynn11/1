# План «каждая лоза доминирует в своих узлах»: считает точки пересечения 5 лоз на торе, группирует близкие узлы,
# распределяет группы по лозам (в группе декор только у одной лозы, у остальных голый стебель) и рисует схемы.
#   python узлы/план_узлов.py --size 2048 --gap 380
import argparse, itertools, json, math
import numpy as np
from PIL import Image, ImageDraw
ap=argparse.ArgumentParser(); ap.add_argument('--size',type=int,default=2048); ap.add_argument('--gap',type=float,default=380)
ap.add_argument('--out',default='узлы'); a=ap.parse_args(); R=a.size
L=dict(A=(1,1,.00,0),B=(1,-1,.40,0),C=(1,2,.15,0),D=(2,1,.62,0),E=(0,1,0,.55))   # p,q,c,x0 (как в сборка_геометрия.py)
COL=dict(A=(40,90,200),B=(210,60,50),C=(40,150,70),D=(235,140,20),E=(140,60,190))
loop={k:R*math.hypot(v[0],v[1]) for k,v in L.items()}
def pos(k,s):
    p,q,c,x0=L[k]; return ((x0+p*s)*R)%R,((c+q*s)*R)%R
nodes=[]
for n1,n2 in itertools.combinations(L,2):
    p1,q1,c1,x1=L[n1]; p2,q2,c2,x2=L[n2]; det=-p1*q2+p2*q1
    for ia in range(-4,5):
        for ib in range(-4,5):
            ra=ia+x2-x1; rb=ib+c2-c1
            s=(-ra*q2+p2*rb)/det; t=(p1*rb-q1*ra)/det
            if 0<=s<1-1e-9 and 0<=t<1-1e-9:
                nodes.append(dict(a=n1,sa=s,b=n2,sb=t,xy=pos(n1,s)))
def td(p,q): 
    dx=abs(p[0]-q[0]); dy=abs(p[1]-q[1]); return math.hypot(min(dx,R-dx),min(dy,R-dy))
# группы узлов, близких на торе (связные компоненты)
g=list(range(len(nodes)))
def f(i):
    while g[i]!=i: g[i]=g[g[i]]; i=g[i]
    return i
for i,j in itertools.combinations(range(len(nodes)),2):
    if td(nodes[i]['xy'],nodes[j]['xy'])<a.gap: g[f(i)]=f(j)
groups={}
for i,n in enumerate(nodes): groups.setdefault(f(i),[]).append(n)
groups=list(groups.values())
# назначение: группа достаётся лозе, которая в ней участвует и пока имеет меньше всего групп; большие группы раньше
cnt={k:0 for k in L}; 
for gr in sorted(groups,key=lambda x:-len(x)):
    cand=sorted({n['a'] for n in gr}|{n['b'] for n in gr},key=lambda k:cnt[k])
    w=cand[0]; cnt[w]+=1
    for n in gr: n['win']=w
    gr[0]['_g']=w
res={k:[] for k in L}
for gr in groups:
    w=gr[0]['win']
    ss=[ (n['sa'] if n['a']==w else n['sb']) for n in gr if w in (n['a'],n['b'])]
    # центр по кругу (петля замкнута): среднее с учётом разрыва
    ss=sorted(ss); 
    if ss[-1]-ss[0]>.5: ss=[s+1 if s<.5 else s for s in ss]
    res[w].append(round((sum(ss)/len(ss))%1,3))
print('групп',len(groups),'узлов',len(nodes)); print({k:v for k,v in res.items()})
json.dump(res,open(a.out+'/доминанты.json','w'),ensure_ascii=False)
# схема тела плитки
S=2; im=Image.new('RGB',(R*S//2,R*S//2),(255,255,255)); d=ImageDraw.Draw(im); sc=S/2
for k in L:
    pts=[pos(k,i/1500) for i in range(1500)]
    for x,y in pts:
        d.ellipse([x*sc-3,y*sc-3,x*sc+3,y*sc+3],fill=COL[k])
for k,ss in res.items():
    for s in ss:
        x,y=pos(k,s)
        for dx in (-R,0,R):
            for dy in (-R,0,R):
                X,Y=(x+dx)*sc,(y+dy)*sc
                d.ellipse([X-a.gap*sc/2,Y-a.gap*sc/2,X+a.gap*sc/2,Y+a.gap*sc/2],outline=COL[k],width=5)
                d.text((X-4,Y-6),k,fill=COL[k])
im.save(a.out+'/схема_доминант.png')
