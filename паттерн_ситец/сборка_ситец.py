# Сборка бесшовного ситца из двух перерисованных веток-лоз (A и B, прозрачный фон) + мелких незабудок-заполнителей.
# Исходный паттерн: две вертикальные лозы рядом, со сдвигом по вертикали. Здесь то же: две колонки, повторяются
# по вертикали с периодом R, по горизонтали — торус (всё, что уходит за край, возвращается с другой стороны).
#
#   python сборка_ситец.py --a ветка_A.png --b ветка_B.png --out раппорт.jpg
import argparse, glob, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True)
ap.add_argument('--small', default='')                # папка с мелочью (*_прозр.png), напр. части/
ap.add_argument('--out', default='раппорт_ситец.jpg')
ap.add_argument('--raport', type=int, default=2175)   # сторона раппорта, px (2175 = 14,5″ при 150 dpi; на 58″ ровно 4 повтора)
ap.add_argument('--fit', type=float, default=1.0)     # высота ветки в долях раппорта (период по вертикали = 1)
ap.add_argument('--ax', type=float, default=0.28)     # центр колонки A по x, доля R
ap.add_argument('--bx', type=float, default=0.74)     # центр колонки B по x, доля R
ap.add_argument('--b-shift', type=float, default=0.5) # сдвиг колонки B по вертикали, доля R
ap.add_argument('--a-shift', type=float, default=0.0)
ap.add_argument('--a-erase', default='0,0.895,0.255,1')   # стереть у ветки A обрезанный краем пион: x0,y0,x1,y1 в долях
ap.add_argument('--b-erase', default='')
ap.add_argument('--bg', default='ECE5D8')
ap.add_argument('--small-cm', type=float, default=2.2)    # размер мелочи на ткани, см (для 150 dpi)
ap.add_argument('--small-max', type=int, default=300)
ap.add_argument('--small-gap', type=float, default=0.7)
ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--quality', type=int, default=95)
a = ap.parse_args()

PX_CM = 150 / 2.54
R = int(a.raport * a.scale)
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


def erase(im, box):
    if not box:
        return im
    x0, y0, x1, y1 = (float(v) for v in box.split(','))
    arr = np.array(im)
    h, w = arr.shape[:2]
    arr[int(y0 * h):int(y1 * h + 1), int(x0 * w):int(x1 * w + 1), 3] = 0
    out = Image.fromarray(arr)
    return out.crop(out.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox())


for key, path, cxf, shf in (('B', a.b, a.bx, a.b_shift), ('A', a.a, a.ax, a.a_shift)):
    im = erase(crop(path), a.a_erase if key == 'A' else a.b_erase)
    k = a.fit * R / im.height                          # высота ветки = fit * R
    el = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    print(f'ветка {key}: {im.size} → {el.size} (масштаб {k:.2f}), центр x={cxf * R:.0f}, сдвиг y={shf * R:.0f}')
    for rep in (-1, 0, 1):                              # соседние копии по вертикали (период R), с возвратом через край
        paste(el, cxf * R, (0.5 + shf) * R + rep * R)

# --- мелочь в пустоты
n_small = 0
if a.small:
    files = sorted(glob.glob(os.path.join(a.small, '*_мелочь_прозр.png')))
    pieces = []
    for f in files:
        im = crop(f)
        w_, h_ = im.size
        if not (95 <= max(w_, h_) <= 150 and 0.88 <= w_ / h_ <= 1.14):     # только целые незабудки, без обрезанных краем половинок
            continue
        k = a.small_cm * PX_CM * a.scale / max(im.size)
        pieces.append(im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS))
    rnd = np.random.default_rng(a.seed)
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
    print(f'мелочи поставлено: {n_small} (целых незабудок в запасе: {len(pieces)})')

canvas.save(a.out, quality=a.quality, subsampling=0)
print('готово', a.out, canvas.size)
t = Image.new('RGB', (R * 2, R * 2))
for x in (0, 1):
    for y in (0, 1):
        t.paste(canvas, (x * R, y * R))
t.resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_плитка2x2.jpg', quality=90)
