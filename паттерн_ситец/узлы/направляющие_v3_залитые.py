# Запасные направляющие для лент схемы v3: кривая СЕРАЯ (зелёная/цветная похожа на стебель, и Gemini её рисует),
# круги ЗАЛИТЫ светло-серым (Gemini крупнит украшения за контур, а залитый круг читается как «поле для рисунка»).
# Геометрия та же, что в узлы/направляющие_v3.py (кривая из схема_v3.json + развёртка в ленту).
#   python -I узлы/направляющие_v3_залитые.py
import json, math
from PIL import Image, ImageDraw
R=2048; ZONE=150
st=json.load(open('узлы/схема_v3.json',encoding='utf8')); PQ=dict(A=(1,1),B=(1,-1),C=(1,2),D=(2,1))
SIZE=dict(A=(2000,660),B=(2000,660),C=(2000,438),D=(2000,438)); NAME=dict(A='A',B='B',C='C',D='D')
for k,prm in st['prm'].items():
    p,q=PQ[k]; loop=R*math.hypot(p,q); W,H=SIZE[k]; kk=loop/W
    ss=sorted(st['s'][k]); gaps=[((ss[(i+1)%len(ss)]-s)%1,s) for i,s in enumerate(ss)]; g,s0=max(gaps); o=(s0+g/2)%1
    f=lambda x: H/2+prm['amp']/kk*math.sin(2*math.pi*prm['cyc']*((x/W+o)%1)+prm['ph'])
    im=Image.new('RGB',(W,H),(255,255,255)); d=ImageDraw.Draw(im)
    for s in ss:
        x=((s-o)%1)*W; r=ZONE/kk; d.ellipse([x-r,f(x)-r,x+r,f(x)+r],fill=(214,214,214))
    d.line([(x,f(x)) for x in range(W)],fill=(110,110,110),width=7)
    im.save(f'узлы/направляющая_v3_{k}_серая_залитая.png'); print(k,W,H)
