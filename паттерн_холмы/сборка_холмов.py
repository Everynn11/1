# Сборка бесшовного паттерна «холмы» (ряды перекрывающихся холмиков, нижний ряд закрывает верхний) на торе.
# Бесшовность: (1) по x холмик, ушедший за край, дорисовывается с другой стороны; (2) по y рисуются три копии плитки
# подряд (верхняя, средняя, нижняя), берётся средняя, поэтому верх следующей плитки правильно закрывает низ предыдущей.
# Расстановка случайная (seed), рядами со сдвигом, с разбросом размера/наклона/цвета, одинаковые холмики не рядом.
#   python -I сборка_холмов.py --elems элементы --w 2175 --h 2175 --cols 9 --out плитка_холмы.png
import argparse, glob, math, os, random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--elems', default='элементы'); ap.add_argument('--out', default='плитка_холмы.png')
ap.add_argument('--w', type=int, default=2175); ap.add_argument('--h', type=int, default=2175)
ap.add_argument('--cols', type=int, default=9)            # холмиков в ряду по ширине плитки
ap.add_argument('--overlap', type=float, default=1.30)    # ширина холмика / шаг по x
ap.add_argument('--rowfrac', type=float, default=0.50)    # шаг рядов / высота холмика (чем меньше, тем плотнее перекрытие)
ap.add_argument('--jx', type=float, default=0.14); ap.add_argument('--jy', type=float, default=0.12)   # разброс положения, доли шага
ap.add_argument('--scale', type=float, default=0.15); ap.add_argument('--rot', type=float, default=5.0)
ap.add_argument('--tone', type=float, default=0.09)       # разброс яркости холмика
ap.add_argument('--bgcol', default='1f4358')              # тёмный фон под всем (щели между холмиками)
ap.add_argument('--wash', type=float, default=0.05); ap.add_argument('--grain', type=float, default=2.0)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
W, H = a.w, a.h
rnd = random.Random(a.seed); nrnd = np.random.default_rng(a.seed)
files = sorted(f for f in glob.glob(os.path.join(a.elems, '*.png')))
lib = [Image.open(f).convert('RGBA') for f in files]
assert lib, 'нет элементов в ' + a.elems
mh = float(np.mean([im.height * (W / a.cols * a.overlap) / im.width for im in lib]))      # средняя высота холмика при целевой ширине
px = W / a.cols
rows = max(2, round(H / (mh * a.rowfrac))); py = H / rows
print(f'элементов {len(lib)}; шаг по x {px:.0f}, рядов {rows}, шаг по y {py:.0f}, средняя высота холмика {mh:.0f} ({mh / py:.1f} шагов)')

# ---- расстановка: (row, x, y_верх, id, scale, rot, flip, tone)
pl = []
used = [0] * len(lib)
def tdist(x1, y1, x2, y2):
    dx = abs(x1 - x2); dx = min(dx, W - dx); dy = abs(y1 - y2); dy = min(dy, H - dy); return math.hypot(dx, dy)
for r in range(rows):
    rowshift = (r % 2) * px / 2 + rnd.uniform(-0.18, 0.18) * px
    for c in range(a.cols):
        x = (c * px + rowshift + rnd.uniform(-a.jx, a.jx) * px) % W
        y = r * py + rnd.uniform(-a.jy, a.jy) * py
        cand = list(range(len(lib))); rnd.shuffle(cand)
        cand.sort(key=lambda i: used[i])                                   # реже использованные вперёд
        pick = None
        for i in cand:
            if all(not (p[3] == i and tdist(x, y % H, p[1], p[2] % H) < 2.6 * px) for p in pl): pick = i; break
        if pick is None: pick = cand[0]
        used[pick] += 1
        pl.append((r, x, y, pick, 1 + rnd.uniform(-a.scale, a.scale), rnd.uniform(-a.rot, a.rot), rnd.random() < 0.5, 1 + rnd.uniform(-a.tone, a.tone)))

def make(p):
    r, x, y, i, sc, rot, fl, tone = p
    im = lib[i]; tw = px * a.overlap * sc; k = tw / im.width
    im = im.resize((max(2, round(im.width * k)), max(2, round(im.height * k * rnd.uniform(0.95, 1.08)))), Image.LANCZOS)
    if fl: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(im).astype(float)
    tint = np.array([tone * rnd.uniform(0.97, 1.03), tone * rnd.uniform(0.97, 1.03), tone * rnd.uniform(0.97, 1.03)])
    arr[..., :3] = np.clip(arr[..., :3] * tint, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8), 'RGBA')
    return im.rotate(rot, expand=True, resample=Image.BICUBIC, center=(im.width / 2, 0))     # поворот вокруг верха холмика

def blit(canvas, im, x0, y0):
    w, h = im.size
    for kx in (-1, 0, 1):
        xx = x0 + kx * W
        if xx >= W or xx + w <= 0 or y0 >= H or y0 + h <= 0: continue
        sx0, sy0 = max(0, -xx), max(0, -y0); sx1, sy1 = min(w, W - xx), min(h, H - y0)
        canvas.alpha_composite(im, (xx + sx0, y0 + sy0), (sx0, sy0, sx1, sy1))

bg = tuple(int(a.bgcol[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
canvas = Image.new('RGBA', (W, H), bg)
made = [make(p) for p in pl]
# порядок: копия -1 (все ряды сверху вниз), копия 0, копия +1; видна только средняя
for kc in (-1, 0, 1):
    for p, im in zip(pl, made):
        x0 = round(p[1] - im.width / 2); y0 = round(p[2] + kc * H)
        blit(canvas, im, x0, y0)
out = np.array(canvas.convert('RGB')).astype(float)

# ---- общий слой: медленные цветовые пятна (периодичны: целые волновые числа) и зерно бумаги (шум на торе)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
if a.wash > 0:
    f = np.zeros((H, W))
    for _ in range(7):
        kx, ky = nrnd.integers(0, 4), nrnd.integers(0, 4)
        if kx == ky == 0: continue
        f += nrnd.uniform(0.5, 1) * np.sin(2 * math.pi * (kx * xx / W + ky * yy / H) + nrnd.uniform(0, 6.28))
    f /= max(1e-6, np.abs(f).max())
    out *= (1 + a.wash * f)[..., None]
if a.grain > 0:
    g = nrnd.normal(0, a.grain, (H, W))
    g = ndi.gaussian_filter(g, 0.8, mode='wrap'); out += g[..., None]
res = np.clip(out, 0, 255).astype(np.uint8); Image.fromarray(res).save(a.out)

# ---- проверка шва на каждой границе повтора: плитка 2×2, перепад на стыке против перепада внутри
big = np.tile(res, (2, 2, 1)).astype(int)
inner = (np.abs(res[:, 1:].astype(int) - res[:, :-1]).mean() + np.abs(res[1:].astype(int) - res[:-1]).mean()) / 2
sx = np.abs(big[:, W] - big[:, W - 1]).mean(); sy = np.abs(big[H] - big[H - 1]).mean()
print('внутри %.2f; шов по x (стык копий) %.2f; шов по y %.2f; край-к-краю %.2f / %.2f' % (inner, sx, sy, np.abs(res[:, 0].astype(int) - res[:, -1]).mean(), np.abs(res[0].astype(int) - res[-1]).mean()))
stem = os.path.splitext(a.out)[0]
t = Image.fromarray(big.astype(np.uint8)); t.resize((min(1600, 2 * W), min(1600, 2 * H)), Image.LANCZOS).save(stem + '_2x2.jpg', quality=90)
c = min(700, W // 2, H // 2)
t.crop((W - c, H - c, W + c, H + c)).save(stem + '_стык_центр.jpg', quality=92)      # полный размер: крест из 4 копий вокруг стыка
