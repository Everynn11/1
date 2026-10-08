# Геометрия «безумной» плитки: прямые и слегка волнистые лозы под разными (замыкающимися на торе) углами + венки.
# Только схема: линии разного цвета. Каждая лоза замкнута на торе, поэтому плитка бесшовна по построению.
#   python геометрия_лоз.py --out геометрия.png --size 2048
import argparse, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='геометрия.png')
ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--width', type=int, default=16)
ap.add_argument('--wave', type=float, default=0.012)      # амплитуда волны, доля размера
ap.add_argument('--wreaths', type=int, default=2)
a = ap.parse_args()

R = a.size
SS = 2
W = R * SS
PALETTE = [(40, 90, 200), (210, 60, 50), (40, 150, 70), (235, 140, 20), (140, 60, 190), (20, 160, 170), (200, 60, 150)]

# (dx, dy) — сколько сторон плитки проходит лоза по x и по y до замыкания, c — смещение, волны на петлю, фаза
LINES = [
    dict(p=1, q=1, c=0.00, waves=3, ph=0.0, name='45° «\\»'),
    dict(p=1, q=-1, c=0.40, waves=3, ph=1.7, name='45° «/»'),
    dict(p=1, q=2, c=0.15, waves=4, ph=0.6, name='63° (1:2)'),
    dict(p=2, q=1, c=0.62, waves=5, ph=2.4, name='27° (2:1)'),
    dict(p=0, q=1, c=0.00, waves=3, ph=0.9, name='вертикаль', x0=0.55),
]


def curve(L, step=1.2):
    p, q = L['p'], L['q']
    length = math.hypot(p * R, q * R)
    n = int(length / step)
    s = np.arange(n) / n
    x = p * R * s + (L.get('x0', 0.0)) * R
    y = q * R * s + L['c'] * R
    tx, ty = p * R, q * R
    nl = math.hypot(tx, ty)
    nx, ny = -ty / nl, tx / nl
    off = a.wave * R * np.sin(2 * np.pi * L['waves'] * s + L['ph'])
    return np.stack([x + nx * off, y + ny * off], 1)


def draw_curve(dr, pts, col, width):
    r = width / 2
    for x, y in pts:
        xm, ym = x % R, y % R
        for dx in (-R, 0, R):
            for dy in (-R, 0, R):
                X, Y = (xm + dx) * SS, (ym + dy) * SS
                if -r * SS <= X <= W + r * SS and -r * SS <= Y <= W + r * SS:
                    dr.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=col)


layers, masks = [], []
for i, L in enumerate(LINES):
    pts = curve(L)
    im = Image.new('L', (W, W), 0)
    draw_curve(ImageDraw.Draw(im), pts, 255, a.width)
    masks.append(np.array(im) > 0)
    layers.append((im, PALETTE[i]))
    print(f"{L['name']}: длина петли {math.hypot(L['p'] * R, L['q'] * R):.0f} px")

# венки: самые пустые места по расстоянию до всех лоз (на торе)
allm = np.zeros((W, W), bool)
for m in masks:
    allm |= m
small = Image.fromarray((allm * 255).astype(np.uint8)).resize((R // 4, R // 4), Image.BOX)
sm = np.array(small) > 0
dist = ndi.distance_transform_edt(~np.tile(sm, (3, 3)))[R // 4:2 * R // 4, R // 4:2 * R // 4] * 4
wr = []
dd = dist.copy()
for k in range(a.wreaths):
    iy, ix = np.unravel_index(np.argmax(dd), dd.shape)
    rad = dd[iy, ix] * 0.72
    wr.append((ix * 4, iy * 4, rad))
    yy, xx = np.mgrid[0:dd.shape[0], 0:dd.shape[1]]
    ddx = np.minimum(np.abs(xx * 4 - ix * 4), R - np.abs(xx * 4 - ix * 4)); ddy = np.minimum(np.abs(yy * 4 - iy * 4), R - np.abs(yy * 4 - iy * 4))
    dd[np.hypot(ddx, ddy) < rad * 1.5] = 0
    print(f'венок {k + 1}: центр ({ix * 4}, {iy * 4}), радиус {rad:.0f} px')
for k, (cx, cy, rad) in enumerate(wr):
    im = Image.new('L', (W, W), 0)
    t = np.linspace(0, 2 * np.pi, int(2 * np.pi * rad / 1.2), endpoint=False)
    pts = np.stack([cx + rad * np.cos(t), cy + rad * np.sin(t) * 1.0], 1)
    draw_curve(ImageDraw.Draw(im), pts, 255, a.width * 0.8)
    masks.append(np.array(im) > 0)
    layers.append((im, PALETTE[(len(LINES) + k) % len(PALETTE)]))

# сборка: полупрозрачно, чтобы видны пересечения
out = Image.new('RGB', (W, W), (255, 255, 255))
for im, col in layers:
    solid = Image.new('RGB', (W, W), col)
    out.paste(solid, (0, 0), im.point(lambda v: 215 if v else 0))
out = out.resize((R, R), Image.LANCZOS)
out.save(a.out)

# число пересечений: связные области, где перекрываются две разные лозы
cross = 0
cnt = np.zeros((W // 2, W // 2), np.uint8)
for m in masks:
    cnt += (np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((W // 2, W // 2), Image.BOX)) > 100).astype(np.uint8)
lab, n = ndi.label(cnt >= 2)
print('пересечений лоз (с венками):', n)
t = Image.new('RGB', (R * 2, R * 2))
for x in (0, 1):
    for y in (0, 1):
        t.paste(out, (x * R, y * R))
t.resize((1600, 1600), Image.LANCZOS).save(a.out.replace('.png', '_2x2.jpg'), quality=90)
print('готово', a.out)
