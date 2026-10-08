# Холсты Printify (Fabric by Yard, Cotton Twill, MWW On Demand, 150 dpi) из одного раппорта 4350×5200.
# Раппорт бесшовный по построению, поэтому холст = раппорт, уложенный плиткой и обрезанный по размеру,
# пиксель в пиксель, без масштабирования (масштаб один на все размеры).
#
#   python холсты_printify.py --raport раппорт/сетка_v8_4350x5200.jpg --mask nofill.jpg --out холсты
#
# --mask — тот же раппорт, собранный с --no-fillers: по нему ищется, где резать (между рядами зайцев).
# Размеры: 29x18 = фэт-квотер, 58x36 = 1 ярд, 58x108 = 3 ярда, 58x180 = 5 ярдов (9x9 не делаем).
import argparse, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SIZES = {'29x18': (4350, 2700),      # фэт-квотер
         '58x36': (8700, 5400),      # 1 ярд
         '58x108': (8700, 16200),    # 3 ярда
         '58x180': (8700, 27000)}    # 5 ярдов

ap = argparse.ArgumentParser()
ap.add_argument('--raport', required=True)
ap.add_argument('--mask', default='')
ap.add_argument('--out', default='холсты')
ap.add_argument('--only', default='')                     # напр. 29x18,58x36
ap.add_argument('--quality', type=int, default=92)
ap.add_argument('--bg', default='ECE5D8')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

arr = np.array(Image.open(a.raport).convert('RGB'))
H, W = arr.shape[:2]
msk = np.array(Image.open(a.mask or a.raport).convert('RGB')).astype(int)
bg = np.array([int(a.bg[i:i + 2], 16) for i in (0, 2, 4)])
busy = (np.abs(msk - bg).max(axis=2) > 14)                 # где что-то нарисовано (без филлеров, если дана маска)
rows, cols = busy.sum(axis=1), busy.sum(axis=0)
x0 = int(np.argmin(cols))                                  # левый/правый край холста там, где столбец пуст
print(f'раппорт {W}×{H}; левый край холста x0={x0} (занято пикселей в столбце: {cols[x0]})')


def best_y0(h):
    cost = rows + np.roll(rows, -(h % H))                  # верхний и нижний края холста
    return int(np.argmin(cost)), int(cost.min())


def canvas(w, h, y0, x0):
    ys, xs = (y0 + np.arange(h)) % H, (x0 + np.arange(w)) % W
    return arr.take(ys, axis=0).take(xs, axis=1)


def save(img, name):
    path = os.path.join(a.out, name)
    Image.fromarray(img).save(path, quality=a.quality, subsampling=0, dpi=(150, 150))
    print(f'  {name}: {img.shape[1]}×{img.shape[0]}, {os.path.getsize(path) / 1e6:.1f} МБ')


want = [s for s in a.only.split(',') if s]
for key, (w, h) in SIZES.items():
    if want and key not in want:
        continue
    y0, c = best_y0(h)
    print(f'{key}: {w}×{h}, верхний край на y0={y0} (занято на краях: {c} px)')
    save(canvas(w, h, y0, x0 if w % W == 0 else 0), f'холст_{key}_{w}x{h}.jpg')
