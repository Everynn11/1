# Направляющие для лент из схемы узлы/схема_v3.json: развёртка каждой лозы в ленту (целая петля), кривая = настоящий изгиб лозы,
# синие круги = места украшений. Левый край ленты сдвинут в самый большой промежуток между кругами.
import json, math
import numpy as np
from PIL import Image, ImageDraw
R=2048; ZONE=150
COL=dict(A=(40,90,220),B=(220,50,50),C=(30,150,60),D=(240,140,10)); NAME=dict(A='синяя',B='красная',C='зелёная',D='оранжевая')
st=json.load(open('узлы/схема_v3.json',encoding='utf8')); PQ=dict(A=(1,1),B=(1,-1),C=(1,2),D=(2,1))
SIZE=dict(A=(2000,660),B=(2000,660),C=(2000,438),D=(2000,438)); out={}
for k,prm in st['prm'].items():
    p,q=PQ[k]; loop=R*math.hypot(p,q); W,H=SIZE[k]; kk=loop/W
    ss=sorted(st['s'][k]); gaps=[((ss[(i+1)%len(ss)]-s)%1,s) for i,s in enumerate(ss)]; g,s0=max(gaps); o=(s0+g/2)%1
    f=lambda x: H/2+prm['amp']/kk*math.sin(2*math.pi*prm['cyc']*((x/W+o)%1)+prm['ph'])
    im=Image.new('RGB',(W,H),(255,255,255)); d=ImageDraw.Draw(im)
    d.line([(x,f(x)) for x in range(W)],fill=COL[k],width=7)
    xs=[]
    for s in ss:
        x=((s-o)%1)*W; xs.append(round(x)); r=ZONE/kk
        d.ellipse([x-r,f(x)-r,x+r,f(x)+r],outline=(0,0,0),width=4)
    im.save(f'узлы/направляющая_v3_{k}_{NAME[k]}.png'); out[k]=dict(ph=o,x=xs,w=W,h=H,amp=prm['amp']/kk,cyc=prm['cyc'],wave_ph=prm['ph'],c=prm['c'],x0=prm['x0'])
    print(k,W,H,'кругов',len(xs),'амплитуда в ленте %.0f px'%(prm['amp']/kk),'радиус круга %.0f px'%(ZONE/kk))
json.dump(out,open('узлы/схема_v3_ленты.json','w'),ensure_ascii=False)
