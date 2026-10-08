# Сборка ситца из набора ветвей (любое число, прозрачный фон) без решётки: жадная расстановка на торе
# со случайным поворотом/зеркалом, допускается небольшое перекрытие краями, остаток добивают незабудки.
#   python сборка_ситец3.py --branches ветки_готовые --small части --out раппорт.jpg --raport 4350 --len 1500
import argparse, glob, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--branches', default='ветки_готовые')
ap.add_argument('--small', default='')
ap.add_argument('--out', default='раппорт_ситец.jpg')
ap.add_argument('--raport', type=int, default=4350)       # сторона раппорта, px (4350 = 29″; 2175 = 14,5″)
ap.add_argument('--len', type=float, default=0.36)        # длинная сторона ветки, доля раппорта
ap.add_argument('--rot', type=float, default=22)          # случайный доворот, ± градусов
ap.add_argument('--overlap', type=float, default=0.10)    # допустимая доля площади ветки, ложащаяся на уже занятое
ap.add_argument('--tries', type=int, default=500)         # кандидатов на каждую новую ветку
ap.add_argument('--max-n', type=int, default=40)
ap.add_argument('--bg', default='ECE5D8')
ap.add_argument('--small-cm', type=float, default=2.2)
ap.add_argument('--small-max', type=int, default=300)
ap.add_argument('--small-gap', type=float, default=0.7)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--quality', type=int, default=95)
a = ap.parse_args()

PX_CM = 150 / 2.54
R = int(a.raport * a.scale)
rnd = np.random.default_rng(a.seed)
canvas = Image.new('RGB', (R, R), tuple(int(a.bg[i:i + 2], 16) for i in (0, 2, 4)))
DF = max(2, R // 400)                                      # масштаб маски занятости
MW = -(-R // DF)
occ = np.zeros((MW, MW), bool)
occ_alpha = Image.new('L', (R, R), 0)


def crop(path):
    im = Image.open(path).convert('RGBA')
    return im.crop(im.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox())


def paste(el, x, y):
    rgb, al = el.convert('RGB'), el.getchannel('A')
    hard = al.point(lambda v: 255 if v > 40 else 0)
    for dx in (-R, 0, R):
        for dy in (-R, 0, R):
            xx, yy = x + dx, y + dy
            if xx < R and yy < R and xx + el.width > 0 and yy + el.height > 0:
                canvas.paste(rgb, (xx, yy), al)
                occ_alpha.paste(255, (xx, yy, xx + el.width, yy + el.height), hard)


def small_mask(el):
    al = np.array(el.getchannel('A')) > 40
    m = np.array(Image.fromarray((al * 255).astype(np.uint8)).resize(
        (max(1, el.width // DF), max(1, el.height // DF)), Image.BILINEAR)) > 40
    return ndi.binary_dilation(m, iterations=1)


files = sorted(glob.glob(os.path.join(a.branches, '*.png')))
base = []
for f in files:
    im = crop(f)
    k = a.len * R / max(im.size)
    base.append((os.path.basename(f), im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)))
    print(f'{os.path.basename(f)}: {im.size} → {base[-1][1].size} ({base[-1][1].width / PX_CM / a.scale:.0f}×{base[-1][1].height / PX_CM / a.scale:.0f} см)')

placed, fails, n = 0, 0, 0
order = []
while placed < a.max_n and fails < 6:
    if not order:
        order = list(range(len(base))); rnd.shuffle(order)
    name, im = base[order.pop()]
    best = None
    for _ in range(a.tries):
        el = im.transpose(Image.FLIP_LEFT_RIGHT) if rnd.random() < 0.5 else im
        el = el.rotate(rnd.uniform(-a.rot, a.rot) + (90 if False else 0), resample=Image.BICUBIC, expand=True)
        m = small_mask(el)
        h, w = m.shape
        cx, cy = rnd.integers(0, MW), rnd.integers(0, MW)
        ys, xs = np.nonzero(m)
        yy, xx = (ys + cy - h // 2) % MW, (xs + cx - w // 2) % MW
        ov = occ[yy, xx].sum() / max(1, len(ys))
        if best is None or ov < best[0]:
            best = (ov, el, cx, cy, yy, xx)
        if ov == 0:
            break
    ov, el, cx, cy, yy, xx = best
    if ov > a.overlap:
        fails += 1
        continue
    fails = 0
    occ[yy, xx] = True
    paste(el, int(cx * DF - el.width / 2), int(cy * DF - el.height / 2))
    placed += 1
print('веток поставлено:', placed)

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
    DF2 = 8
    MW2 = -(-R // DF2)
    margin = a.small_gap * PX_CM * a.scale / DF2
    while n_small < a.small_max and pieces:
        o2 = np.array(occ_alpha.resize((MW2, MW2), Image.BOX)) > 8
        dist = ndi.distance_transform_edt(~np.tile(o2, (3, 3)))[MW2:2 * MW2, MW2:2 * MW2]
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        bd = dist[cy, cx]
        cand = [p for p in pieces if 0.5 * math.hypot(*p.size) / DF2 + margin <= bd]
        if not cand:
            break
        pc = cand[rnd.integers(len(cand))].rotate(rnd.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        paste(pc, int(cx * DF2 + DF2 / 2 - pc.width / 2), int(cy * DF2 + DF2 / 2 - pc.height / 2))
        n_small += 1
    print('мелочи поставлено:', n_small)

canvas.save(a.out, quality=a.quality, subsampling=0)
t = Image.new('RGB', (R * 2, R * 2))
for x in (0, 1):
    for y in (0, 1):
        t.paste(canvas, (x * R, y * R))
t.resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_плитка2x2.jpg', quality=90)
print('готово', a.out, canvas.size)
