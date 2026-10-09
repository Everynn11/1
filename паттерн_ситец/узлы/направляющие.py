# Направляющие для лент «каждая лоза доминирует в своих узлах». Читает узлы/доминанты.json.
# Для каждой лозы: лента = целая петля (n=1). Красная волна — путь стебля, цветные круги — где ставить украшения.
# Левый/правый край ленты сдвинут (ph) в самый большой промежуток между кругами, поэтому у краёв голый стебель.
import json, math, os
import numpy as np
from PIL import Image, ImageDraw
R=2048; ZONE=150
loop=dict(A=R*math.sqrt(2),B=R*math.sqrt(2),C=R*math.sqrt(5),D=R*math.sqrt(5),E=R)
size=dict(A=(2000,660),B=(2000,660),C=(2000,450),D=(2000,450),E=(2000,660))
waves=dict(A=(2,0.3),B=(2.5,1.1),C=(3,0.0),D=(3.5,2.0),E=(2,0.9))   # целых периодов нет у 2.5/3.5: подбираем ниже
D=json.load(open('узлы/доминанты.json'))
info={}
for k,ss in D.items():
    ss=sorted(ss); gaps=[((ss[(i+1)%len(ss)]-s)%1, s) for i,s in enumerate(ss)]
    g,s0=max(gaps); o=(s0+g/2)%1                      # середина самого большого промежутка
    W,H=size[k]; cyc=round(waves[k][0]); ph0=waves[k][1]
    yc=H/2; amp=H*0.12
    im=Image.new('RGB',(W,H),(255,255,255)); d=ImageDraw.Draw(im)
    pts=[(x,yc+amp*math.sin(2*math.pi*cyc*x/W+ph0)) for x in range(W)]
    d.line(pts,fill=(232,40,45),width=7)
    xs=[]
    for s in ss:
        x=((s-o)%1)*W; xs.append(round(x))
        y=yc+amp*math.sin(2*math.pi*cyc*x/W+ph0); r=ZONE/loop[k]*W
        d.ellipse([x-r,y-r,x+r,y+r],outline=(40,90,220),width=5)
    im.save(f'узлы/направляющая_{k}.png')
    info[k]=dict(ph=round(o,4),x=xs,w=W,h=H,cycles=cyc,phase=ph0,amp=amp)
    print(k,info[k])
json.dump(info,open('узлы/ленты_параметры.json','w'),ensure_ascii=False)
