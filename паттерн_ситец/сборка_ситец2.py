# Сборка ситца из ЦЕЛЫХ веток (A и B на прозрачном фоне): шахматка из зеркальных копий на торе + незабудки в пустоты.
#   python сборка_ситец2.py --a ветки_готовые/ветка_A.png --b ветки_готовые/ветка_B.png --small части --out раппорт.jpg
import argparse, glob, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True)
ap.add_argument('--small', default='')
ap.add_argument('--out', default='раппорт_ситец.jpg')
ap.add_argument('--raport', type=int, default=2175)
ap.add_argument('--h', type=float, default=0.56)          # высота ветки, доля раппорта
ap.add_argument('--cols', type=int, default=2)            # колонок в раппорте
ap.add_argument('--rows', type=int, default=2)            # веток в колонке
ap.add_argument('--tilt', type=float, default=0.0)        # случайный доворот ветки, градусы (±)
ap.add_argument('--jitter', type=float, default=0.0)      # случайный сдвиг, доля шага
ap.add_argument('--bg', default='ECE5D8')
ap.add_argument('--small-cm', type=float, default=2.2)
ap.add_argument('--small-max', type=int, default=300)
ap.add_argument('--small-gap', type=float, default=0.7)
ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--quality', type=int, default=95)
a = ap.parse_args()

PX_CM = 150 / 2.54
R = int(a.raport * a.scale)
rnd = np.random.default_rng(a.seed)
canvas = Image.new('RGB', (R, R), tuple(int(a.bg[i:i + 2], 16) for i in (0, 2, 4)))
occ_alpha = Image.new('L', (R, R), 0)


def crop(path):
    im = Image.open(path).convert('RGBA')
    return im.crop(im.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox())


def paste(el, cx, cy):
    x, y = int(round(cx - el.width / 2)), int(round(cy - el.height / 2))
    rgb, al = el.convert('RGB'), el.getchannel('A')
    hard = al.point(lambda v: 255 if v > 40 else 0)
    for dx in (-R, 0, R):
        for dy in (-R, 0, R):
            xx, yy = x + dx, y + dy
            if xx < R and yy < R and xx + el.width > 0 and yy + el.height > 0:
                canvas.paste(rgb, (xx, yy), al)
                occ_alpha.paste(255, (xx, yy, xx + el.width, yy + el.height), hard)


base = {}
for key, path in (('A', a.a), ('B', a.b)):
    im = crop(path)
    k = a.h * R / im.height
    base[key] = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    print(f'ветка {key}: {im.size} → {base[key].size} (масштаб {k:.2f}), на ткани {base[key].width / PX_CM / a.scale:.1f}×{base[key].height / PX_CM / a.scale:.1f} см')

# шахматка: колонка c, ряд r; колонки сдвинуты по вертикали через одну; чередуем A/B и зеркалим по диагоналям
seq = []
dy = R / a.rows
for c in range(a.cols):
    for r in range(a.rows):
        cx = (c + 0.5) * R / a.cols
        cy = (r + 0.5) * dy + (0.5 * dy if c % 2 else 0)
        name = 'A' if (c + r) % 2 == 0 else 'B'
        flip = ((c // 1 + r) // 1) % 2 == 1 if False else ((c + 2 * r) % 4 >= 2)
        seq.append((name, flip, cx, cy))
for name, flip, cx, cy in seq:
    el = base[name].transpose(Image.FLIP_LEFT_RIGHT) if flip else base[name]
    if a.tilt:
        el = el.rotate(rnd.uniform(-a.tilt, a.tilt), resample=Image.BICUBIC, expand=True)
    if a.jitter:
        cx += rnd.uniform(-a.jitter, a.jitter) * R / a.cols
        cy += rnd.uniform(-a.jitter, a.jitter) * dy
    paste(el, cx, cy)

# незабудки в пустоты
n_small = 0
if a.small:
    pieces = []
    for f in sorted(glob.glob(os.path.join(a.small, '*_мелочь_прозр.png'))):
        im = crop(f)
        w_, h_ = im.size
        if not (95 <= max(w_, h_) <= 150 and 0.88 <= w_ / h_ <= 1.14):
            continue
        k = a.small_cm * PX_CM * a.scale / max(im.size)
        pieces.append(im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS))
    DF = 6
    MW = -(-R // DF)
    margin = a.small_gap * PX_CM * a.scale / DF
    while n_small < a.small_max and pieces:
        occ = np.array(occ_alpha.resize((MW, MW), Image.BOX)) > 8
        dist = ndi.distance_transform_edt(~np.tile(occ, (3, 3)))[MW:2 * MW, MW:2 * MW]
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        best = dist[cy, cx]
        cand = [p for p in pieces if 0.5 * math.hypot(*p.size) / DF + margin <= best]
        if not cand:
            break
        pc = cand[rnd.integers(len(cand))].rotate(rnd.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        paste(pc, cx * DF + DF / 2, cy * DF + DF / 2)
        n_small += 1
    print(f'мелочи поставлено: {n_small}')

canvas.save(a.out, quality=a.quality, subsampling=0)
t = Image.new('RGB', (R * 2, R * 2))
for x in (0, 1):
    for y in (0, 1):
        t.paste(canvas, (x * R, y * R))
t.resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_плитка2x2.jpg', quality=90)
print('готово', a.out, canvas.size)
