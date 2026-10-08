# Плитка ситца из ДВУХ лент-бордюров (каждая нарисована Gemini, повторяется по горизонтали).
# Четыре лозы идут по диагоналям на торе: две «\» и две «/». На каждой лозе своя лента (чередуются) и свой сдвиг по фазе.
# В узлах пересечения «\» и «/» поочерёдно одна проходит над другой (поднимается только сама лоза, не цветы и листья).
#   python сборка_из_лент.py --r1 лента_1.png --r2 лента_2.png --out плитка.png --size 2048
import argparse, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--r1', required=True)
ap.add_argument('--r2', required=True)
ap.add_argument('--out', default='плитка_из_лент.png')
ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--bg', default='')
ap.add_argument('--cut', type=int, default=64)            # радиус «над/под» вокруг узла, px
ap.add_argument('--tol', type=int, default=38)            # допуск цвета лозы
ap.add_argument('--phases', default='0,0.27,0.52,0.77')   # сдвиг по фазе для четырёх лоз, доля периода
ap.add_argument('--nrep', type=int, default=3)
a = ap.parse_args()

R = a.size
n = 2                                                     # периодов на ось
T = R / n * math.sqrt(2)                                  # период вдоль диагонали
ph = [float(x) for x in a.phases.split(',')]


def load_ribbon(path):
    im = Image.open(path).convert('RGBA')
    W0, H0 = im.size
    arr = np.array(im)
    vine = arr[:, 0][arr[:, 0, 3] > 200][:, :3].mean(axis=0)
    strip = Image.new('RGBA', (W0 * a.nrep, H0), (0, 0, 0, 0))
    for i in range(a.nrep):
        strip.alpha_composite(im, (i * W0, 0))
    k = T / W0
    strip = strip.resize((int(round(W0 * a.nrep * k)), int(round(H0 * k))), Image.LANCZOS)
    return strip, vine


rib = {1: load_ribbon(a.r1), 2: load_ribbon(a.r2)}
print('ленты:', {k: (v[0].size, v[1].astype(int)) for k, v in rib.items()})

# четыре оси: направление (+1 «\» вниз-вправо; -1 «/» вверх-вправо), смещение c, лента, зеркало по вертикали
AXES = [dict(dir=+1, c=0.00 * R, r=1, flip=False),
        dict(dir=+1, c=0.50 * R, r=2, flip=False),
        dict(dir=-1, c=0.25 * R, r=2, flip=True),
        dict(dir=-1, c=0.75 * R, r=1, flip=True)]


def paste_wrap(dst, im, cx, cy):
    """Вставка с обёрткой по тору, im может быть больше dst."""
    w, h = im.size
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    for kx in range(-3, 4):
        for ky in range(-3, 4):
            xx, yy = x0 + kx * R, y0 + ky * R
            if xx >= R or yy >= R or xx + w <= 0 or yy + h <= 0:
                continue
            sx0, sy0 = max(0, -xx), max(0, -yy)
            sx1, sy1 = min(w, R - xx), min(h, R - yy)
            dst.alpha_composite(im, (xx + sx0, yy + sy0), (sx0, sy0, sx1, sy1))


axis_layers = []
for i, ax in enumerate(AXES):
    strip, vine = rib[ax['r']]
    s = strip.transpose(Image.FLIP_TOP_BOTTOM) if ax['flip'] else strip
    ang = -45 if ax['dir'] > 0 else 45
    rs = s.rotate(ang, resample=Image.BICUBIC, expand=True)
    # центр полосы на оси: середина n периодов, сдвинутая по фазе
    u = (n / 2 + ph[i]) * T
    d = np.array([1.0, ax['dir']]) / math.sqrt(2)
    p = (np.array([0.0, ax['c']]) + d * u) % R
    L = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    paste_wrap(L, rs, p[0], p[1])
    axis_layers.append((ax, L, vine))

canvas = Image.new('RGBA', (R, R), (0, 0, 0, 0))
for ax, L, vine in axis_layers:
    if ax['dir'] > 0:
        canvas.alpha_composite(L)
for ax, L, vine in axis_layers:
    if ax['dir'] < 0:
        canvas.alpha_composite(L)

# узлы пересечения осей «\» и «/»
nodes = []
for A in AXES:
    for B in AXES:
        if A['dir'] != +1 or B['dir'] != -1:
            continue
        for kx in range(-4, 5):
            x = (B['c'] - A['c']) / 2 + kx * R / 2
            y = x + A['c']
            q = (x % R, y % R)
            if all(math.hypot(min(abs(q[0] - r[0]), R - abs(q[0] - r[0])), min(abs(q[1] - r[1]), R - abs(q[1] - r[1]))) > 40 for r in nodes):
                nodes.append(q)
print('узлов:', len(nodes))

yy, xx = np.mgrid[0:R, 0:R]
nodes.sort(key=lambda p: (round(p[1] / 120), p[0]))
for i, (x, y) in enumerate(nodes):
    if i % 2:
        continue                                          # в чётных узлах «\» поверх, в нечётных остаётся «/» поверх
    dx = np.minimum(np.abs(xx - x), R - np.abs(xx - x)); dy = np.minimum(np.abs(yy - y), R - np.abs(yy - y))
    near = np.hypot(dx, dy) < a.cut
    for ax, L, vine in axis_layers:
        if ax['dir'] < 0:
            continue
        arr = np.array(L)
        d_ = np.abs(arr[..., :3].astype(int) - vine[None, None, :]).max(axis=2)
        m = (d_ < a.tol) & (arr[..., 3] > 200) & near
        if m.sum() == 0:
            continue
        m = ndi.binary_dilation(m, iterations=2) & (arr[..., 3] > 40) & near
        patch = arr.copy(); patch[..., 3] = np.where(m, patch[..., 3], 0)
        canvas.alpha_composite(Image.fromarray(patch))

bgc = tuple(int(a.bg[i:i + 2], 16) for i in (0, 2, 4)) + (255,) if a.bg else (255, 255, 255, 255)
out = Image.new('RGBA', (R, R), bgc)
out.alpha_composite(canvas)
out.convert('RGB').save(a.out)
canvas.save(os.path.splitext(a.out)[0] + '_прозр.png')
t = Image.new('RGB', (R * 2, R * 2))
for x_ in (0, 1):
    for y_ in (0, 1):
        t.paste(out.convert('RGB'), (x_ * R, y_ * R))
t.save(os.path.splitext(a.out)[0] + '_2x2.png')
t.resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_2x2_превью.jpg', quality=90)
o = np.array(out.convert('RGB')).astype(float)
inner = (np.abs(o[:, 1:] - o[:, :-1]).mean() + np.abs(o[1:] - o[:-1]).mean()) / 2
print('перепад внутри %.2f; стык лево-право %.2f, верх-низ %.2f' % (inner, np.abs(o[:, -1] - o[:, 0]).mean(), np.abs(o[-1] - o[0]).mean()))
print('готово', a.out)
