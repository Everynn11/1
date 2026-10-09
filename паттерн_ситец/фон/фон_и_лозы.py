# Пример «два слоя»: бледный мелкий фон с шагом R/4 (бесшовен, т.к. 4 повтора на плитку) + 2 крупные лозы (A листья, B шиповник).
#   python -I фон/фон_и_лозы.py --fg фон/перед_AB_прозр.png --out фон/пример_2_слоя.png
import argparse, random, math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ap=argparse.ArgumentParser(); ap.add_argument('--fg',default='фон/перед_AB_прозр.png'); ap.add_argument('--out',default='фон/пример_2_слоя.png')
ap.add_argument('--bg',default='F8F3EA'); ap.add_argument('--tint',type=float,default=0.28); ap.add_argument('--n',type=int,default=4); a=ap.parse_args()
random.seed(3); R=2048; P=R//a.n
bgc=tuple(int(a.bg[i:i+2],16) for i in (0,2,4))
def comps(path,minarea,maxarea):
    im=Image.open(path).convert('RGBA'); arr=np.array(im); al=arr[...,3]>200
    # убрать стебель: самая большая компонента (лоза + всё, что к ней приросло по цвету) не нужна — берём отдельные «острова» украшений
    thin=ndi.binary_opening(al,iterations=9)               # стебель тоньше 18 px исчезает, остаются широкие листья/цветы
    lab,n=ndi.label(thin); out=[]
    for i in range(1,n+1):
        m=lab==i; ar=m.sum()
        if minarea<ar<maxarea:
            m=ndi.binary_dilation(m,iterations=9)&al
            ys,xs=np.nonzero(m); crop=arr[ys.min():ys.max()+1,xs.min():xs.max()+1].copy()
            crop[...,3]=np.where(m[ys.min():ys.max()+1,xs.min():xs.max()+1],crop[...,3],0); out.append(Image.fromarray(crop))
    return out
leaves=comps('лента/схема3_A.png',8000,10**7); flowers=comps('лента/схема3_B.png',8000,10**7)
print('листьев',len(leaves),'цветов',len(flowers))
def tint(im,s):
    arr=np.array(im).astype(float); arr[...,:3]=arr[...,:3]*s+np.array(bgc)*(1-s); return Image.fromarray(arr.astype(np.uint8))
def fit(im,h): return im.resize((max(1,int(im.width*h/im.height)),h),Image.LANCZOS)
tile=Image.new('RGBA',(P,P),bgc+(255,))
spots=[(0.25,0.25,leaves,0.50,18),(0.75,0.25,flowers,0.40,0),(0.25,0.75,flowers,0.40,40),(0.75,0.75,leaves,0.50,-25)]
for fx,fy,pool,hh,rot in spots:
    m=tint(random.choice(pool),a.tint); m=fit(m,int(P*hh)).rotate(rot,expand=True,resample=Image.BICUBIC)
    for ox in (-P,0,P):                                        # обёртка по тору: часть, ушедшая за край, возвращается с другой стороны
        for oy in (-P,0,P):
            tile.paste(m,(int(fx*P-m.width/2)+ox,int(fy*P-m.height/2)+oy),m)
bg=Image.new('RGBA',(R,R))
for x in range(a.n):
    for y in range(a.n): bg.paste(tile,(x*P,y*P))
fg=Image.open(a.fg).convert('RGBA'); out=bg.copy(); out.alpha_composite(fg); out=out.convert('RGB'); out.save(a.out)
t=Image.new('RGB',(R*2,R*2))
for x in (0,1):
    for y in (0,1): t.paste(out,(x*R,y*R))
t.resize((1600,1600),Image.LANCZOS).save(a.out.replace('.png','_2x2.jpg'),quality=90)
o=np.array(out).astype(float); inner=(np.abs(o[:,1:]-o[:,:-1]).mean()+np.abs(o[1:]-o[:-1]).mean())/2
# швы фона: разница на границах под-плиток (каждые P px) против обычных соседних столбцов/строк
def seam(o,P):
    cols=[np.abs(o[:,x]-o[:,x-1]).mean() for x in range(P,R,P)]; rows=[np.abs(o[y]-o[y-1]).mean() for y in range(P,R,P)]
    return np.mean(cols),np.mean(rows)
bgo=np.array(bg.convert('RGB')).astype(float); print('ФОН отдельно: на границах под-плиток %.2f / %.2f, внутри %.2f'%(*seam(bgo,P),(np.abs(bgo[:,1:]-bgo[:,:-1]).mean()+np.abs(bgo[1:]-bgo[:-1]).mean())/2))
print('внутри %.2f; лево-право %.2f; верх-низ %.2f'%(inner,np.abs(o[:,-1]-o[:,0]).mean(),np.abs(o[-1]-o[0]).mean()))
