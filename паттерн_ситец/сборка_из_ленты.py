# Плитка ситца из ОДНОЙ ленты-бордюра (повторяется по горизонтали, нарисована Gemini).
# Лента кладётся вдоль четырёх диагональных осей (две «\», две «/») на торе, повторяется вдоль оси встык,
# в узлах пересечения лозы поочерёдно проходят одна над другой.
#   python сборка_из_ленты.py --ribbon лента.webp --out плитка.png --size 2048
import argparse, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--ribbon', required=True)
ap.add_argument('--out', default='плитка_из_ленты.png')
ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--bg', default='')                      # HEX фона; пусто = прозрачный/белый
ap.add_argument('--scale', type=float, default=1.0)      # множитель размера ленты (1 = период по оси ровно R·√2/2)
ap.add_argument('--cut', type=int, default=62)           # радиус «над/под» вокруг узла, px
ap.add_argument('--tol', type=int, default=26)           # допуск цвета лозы
ap.add_argument('--flip-b', action='store_true')         # зеркалить ленту «/» (по вертикали)
a = ap.parse_args()

R = a.size
rib0 = Image.open(a.ribbon).convert('RGBA')
Wr, Hr = rib0.size
NREP = 4
rib = Image.new('RGBA', (Wr * NREP, Hr), (0, 0, 0, 0))
for i_ in range(NREP):
    rib.alpha_composite(rib0, (i_ * Wr, 0))
n = 2                                                    # периодов на сторону плитки
T = R / n * math.sqrt(2)                                 # длина периода вдоль диагонали
k = T / Wr * a.scale
rib = rib.resize((int(round(Wr * NREP * k)), int(round(Hr * k))), Image.LANCZOS)
print(f'лента {Wr}×{Hr} ×{NREP} периода → {rib.size} (масштаб {k:.3f}), период по оси {T:.0f} px')

# цвет лозы (по краю ленты)
arr = np.array(rib0)
col = arr[:, 0][arr[:, 0, 3] > 200][:, :3].mean(axis=0)
print('цвет лозы', col.astype(int))

canvas = Image.new('RGBA', (R, R), (0, 0, 0, 0))


def paste_wrap(dst, im, cx, cy):
    x, y = int(round(cx - im.width / 2)), int(round(cy - im.height / 2))
    for dx in (-R, 0, R):
        for dy in (-R, 0, R):
            xx, yy = x + dx, y + dy
            if xx < R and yy < R and xx + im.width > 0 and yy + im.height > 0:
                dst.alpha_composite(im, (max(0, xx), max(0, yy)), (max(0, -xx), max(0, -yy), min(im.width, R - xx), min(im.height, R - yy)))


# четыре оси: (направление, смещение c)
AXES = []
for c in (0.0, 0.5):
    AXES.append(dict(dir=+1, c=c * R))                    # y = x + c   (экран: вниз-вправо)
for c in (0.25, 0.75):
    AXES.append(dict(dir=-1, c=c * R))                    # y = -x + c  (экран: вверх-вправо)

rot = {+1: rib.rotate(-45, resample=Image.BICUBIC, expand=True),
       -1: (rib.transpose(Image.FLIP_TOP_BOTTOM) if a.flip_b else rib).rotate(45, resample=Image.BICUBIC, expand=True)}


def axis_points(ax, m):
    """Центр m-го периода на оси."""
    u = (m + 0.5) * T
    d = np.array([1.0, ax['dir']]) / math.sqrt(2)
    p = np.array([0.0, ax['c'] % R]) + d * u
    return p % R


layers = {+1: Image.new('RGBA', (R, R), (0, 0, 0, 0)), -1: Image.new('RGBA', (R, R), (0, 0, 0, 0))}
for ax in AXES:
    L = layers[ax['dir']]
    cx, cy = axis_points(ax, (n - 1) / 2)             # центр оси (полоса длиннее оси, лишние периоды ложатся на те же места)
    paste_wrap(L, rot[ax['dir']], cx, cy)

# узлы пересечений осей
nodes = []
for A in AXES:
    for B in AXES:
        if A['dir'] != +1 or B['dir'] != -1:
            continue
        # y = x + cA ; y = -x + cB  → x = (cB - cA)/2 (mod R/2 шагами)
        for kx in range(-4, 5):
            x = (B['c'] - A['c']) / 2 + kx * R / 2
            y = x + A['c']
            if 0 <= x < R and 0 <= y < R:
                nodes.append((x, y))
            elif 0 <= x % R < R and 0 <= y % R < R and (x % R, y % R) not in nodes:
                nodes.append((x % R, y % R))
uniq = []
for p in nodes:
    if all(math.hypot(min(abs(p[0] - q[0]), R - abs(p[0] - q[0])), min(abs(p[1] - q[1]), R - abs(p[1] - q[1]))) > 30 for q in uniq):
        uniq.append(p)
nodes = uniq
print('узлов:', len(nodes))

# базовый порядок: «\» (dir=+1) снизу, «/» (dir=-1) сверху
canvas.alpha_composite(layers[+1])
canvas.alpha_composite(layers[-1])

# над/под: в каждом втором узле поднимаем «\» над «/» только по лозе вокруг узла
vine_mask_full = np.array(layers[+1])
d = np.abs(vine_mask_full[..., :3].astype(int) - col[None, None, :]).max(axis=2)
vine_px = (d < a.tol) & (vine_mask_full[..., 3] > 200)
yy, xx = np.mgrid[0:R, 0:R]
nodes.sort(key=lambda p: (round(p[1] / 100), p[0]))
for i, (x, y) in enumerate(nodes):
    if i % 2:
        continue
    dx = np.minimum(np.abs(xx - x), R - np.abs(xx - x)); dy = np.minimum(np.abs(yy - y), R - np.abs(yy - y))
    m = vine_px & (np.hypot(dx, dy) < a.cut)
    m = ndi.binary_dilation(m, iterations=2) & (vine_mask_full[..., 3] > 40)
    patch = np.array(layers[+1]); patch[..., 3] = np.where(m, patch[..., 3], 0)
    canvas.alpha_composite(Image.fromarray(patch))

if a.bg:
    bgc = tuple(int(a.bg[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    out = Image.new('RGBA', (R, R), bgc); out.alpha_composite(canvas)
else:
    out = Image.new('RGBA', (R, R), (255, 255, 255, 255)); out.alpha_composite(canvas)
out.convert('RGB').save(a.out)
canvas.save(os.path.splitext(a.out)[0] + '_прозр.png')
t = Image.new('RGB', (R * 2, R * 2))
for x_ in (0, 1):
    for y_ in (0, 1):
        t.paste(out.convert('RGB'), (x_ * R, y_ * R))
t.resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_плитка2x2.jpg', quality=90)
print('готово', a.out)
