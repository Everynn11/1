# Холсты Printify (Fabric by Yard, Cotton Twill, 150 dpi) из бесшовной плитки ситца 2175×2175 (2175 = 4350/2, делит 8700 нацело).
# Плитка бесшовна по построению, поэтому холст = плитка, уложенная и обрезанная по размеру, пиксель в пиксель, без масштабирования.
# Каждая граница повтора внутри холста проверяется: перепад на ней сравнивается с перепадом внутри плитки.
#   python -I холсты_ситец.py --tile фон/ситец_2слоя.png --out холсты [--only 29x18,58x36]
import argparse, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SIZES = {'29x18': (4350, 2700), '58x36': (8700, 5400), '58x108': (8700, 16200), '58x180': (8700, 27000)}
ap = argparse.ArgumentParser()
ap.add_argument('--tile', required=True); ap.add_argument('--out', default='холсты'); ap.add_argument('--only', default='')
ap.add_argument('--quality', type=int, default=90)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tile = np.array(Image.open(a.tile).convert('RGB')); H, W = tile.shape[:2]
assert H == W and 8700 % W == 0, f'плитка {W}×{H}: нужен квадрат, ширина делит 8700'
d = lambda a_, b_: float(np.abs(a_.astype(int) - b_.astype(int)).mean())
inner = (d(tile[:, 1:], tile[:, :-1]) + d(tile[1:], tile[:-1])) / 2
print(f'плитка {W}×{H}; перепад внутри {inner:.2f}')
for key, (w, h) in SIZES.items():
    if a.only and key not in a.only.split(','): continue
    nx, ny = -(-w // W), -(-h // H)
    img = np.tile(tile, (ny, nx, 1))[:h, :w]
    cols = [d(img[:, x], img[:, x - 1]) for x in range(W, w, W)]; rows = [d(img[y], img[y - 1]) for y in range(H, h, H)]
    cm, rm = (max(cols) if cols else 0.0), (max(rows) if rows else 0.0)
    print(f'{key}: {w}×{h}; границ повтора по x {len(cols)} (макс. перепад {cm:.2f}), по y {len(rows)} (макс. {rm:.2f}); внутри {inner:.2f}')
    assert max(cm, rm) < inner * 1.5 + 0.5, 'шов заметнее внутреннего перепада'
    path = os.path.join(a.out, f'холст_{key}_{w}x{h}.jpg')
    Image.fromarray(img).save(path, quality=a.quality, subsampling=0, dpi=(150, 150))
    print(f'  {path}: {os.path.getsize(path) / 1e6:.1f} МБ')
    del img
