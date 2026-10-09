# Снимает ПРИМЕРНЫЕ кривые дуг холмиков с картинки-образца (только геометрия: тонкие линии, не пиксели рисунка).
# Светлые линии под дугами (каждая отдельная, по одной на холмик) -> компоненты -> скелет -> упорядоченная кривая -> фильтр «похоже на дугу».
# (Тёмные дуги слиплись в сетку и на сегменты не режутся.)
#   python -I снять_дуги.py --ref образец.webp --out дуги.json --preview дуги_превью.png
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
ap = argparse.ArgumentParser(); ap.add_argument('--ref', required=True); ap.add_argument('--out', default='дуги.json'); ap.add_argument('--preview', default='дуги_превью.png')
ap.add_argument('--light', type=float, default=205); ap.add_argument('--minlen', type=int, default=70); ap.add_argument('--minw', type=int, default=70)
a = ap.parse_args()
im = Image.open(a.ref).convert('RGB'); arr = np.array(im).astype(float)
H, W = arr.shape[:2]
box = (125, 0, 1080, 880)                                              # полотно образца без полей и свитка (для этого образца)
lum = arr.mean(axis=2)
light = np.zeros((H, W), bool); light[box[1]:box[3], box[0]:box[2]] = lum[box[1]:box[3], box[0]:box[2]] > a.light
light = ndi.binary_opening(light, iterations=1)
lab, n = ndi.label(light, structure=np.ones((3, 3)))
curves = []
for i in range(1, n + 1):
    m = lab == i
    if m.sum() < a.minlen: continue
    sk = skeletonize(m); ys, xs = np.nonzero(sk)
    if len(xs) < a.minlen * 0.6: continue
    o = np.argsort(xs); xs, ys = xs[o].astype(float), ys[o].astype(float)
    wid = xs[-1] - xs[0]
    if wid < a.minw: continue
    k = 11; yk = np.convolve(np.pad(ys, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
    ch = yk[0] + (yk[-1] - yk[0]) * (xs - xs[0]) / max(1, wid)
    rise = float(np.mean(ch - yk))                                     # >0: середина выше хорды (выпуклая вверх)
    if rise < 0.05 * wid or rise > 0.45 * wid: continue
    curves.append(dict(x=xs.tolist(), y=yk.tolist(), w=float(wid), rise=rise))
print(f'компонентов {n}, дуг похожих на холмик: {len(curves)}')
json.dump(curves, open(a.out, 'w'))
pv = Image.fromarray(arr.astype(np.uint8)).copy(); d = ImageDraw.Draw(pv)
for c in curves: d.line(list(zip(c['x'], c['y'])), fill=(255, 40, 40), width=3)
pv.save(a.preview)
