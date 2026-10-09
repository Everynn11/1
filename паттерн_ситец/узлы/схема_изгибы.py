# Схема плитки с НАСТОЯЩИМИ изгибами лоз (та же волна, что на направляющих) и кругами-украшениями.
#   python -I узлы/схема_изгибы.py  → узлы/схема_изгибы.png
import json, math
import numpy as np
from PIL import Image, ImageDraw
R=2048; SS=2; W=R*SS
LINES=dict(A=(1,1,.00,0),B=(1,-1,.40,0),C=(1,2,.15,0),D=(2,1,.62,0),E=(0,1,0,.55))
COL=dict(A=(40,90,220),B=(220,50,50),C=(30,150,60),D=(240,140,10),E=(150,50,200))
P=json.load(open('узлы/ленты_параметры.json',encoding='utf8'))
im=Image.new('RGB',(W,W),(255,255,255)); d=ImageDraw.Draw(im)
def pt(k,s):
    p,q,c,x0=LINES[k]; n=math.hypot(p,q); loop=R*n; dx,dy=p/n,q/n
    prm=P[k]; Wr=prm['w']; kk=loop/Wr
    x=((s-prm['ph'])%1)*Wr
    off=prm['amp']*math.sin(2*math.pi*prm['cycles']*x/Wr+prm['phase'])*kk
    return ((x0+p*s)*R-dy*off)%R,((c+q*s)*R+dx*off)%R
for k in LINES:
    n=4000; pts=[pt(k,i/n) for i in range(n)]
    for x,y in pts:
        for ox in (-R,0,R):
            for oy in (-R,0,R):
                X,Y=(x+ox)*SS,(y+oy)*SS
                if -20<=X<=W+20 and -20<=Y<=W+20: d.ellipse([X-7,Y-7,X+7,Y+7],fill=COL[k])
    prm=P[k]; loop=R*math.hypot(*LINES[k][:2]); r=150
    for xr in prm['x']:
        s=(xr/prm['w']+prm['ph'])%1; x,y=pt(k,s)
        for ox in (-R,0,R):
            for oy in (-R,0,R):
                X,Y=(x+ox)*SS,(y+oy)*SS
                d.ellipse([X-r*SS,Y-r*SS,X+r*SS,Y+r*SS],outline=COL[k],width=8)
im.resize((R,R),Image.LANCZOS).save('узлы/схема_изгибы.png')
print('готово')
