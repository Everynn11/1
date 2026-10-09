# Заглушки для проверки конвейера (НЕ стиль паттерна): рисует лист из 9 схематичных холмиков на белом фоне.
#   python -I тест_заглушки.py --out /tmp/лист_тест.png
import argparse, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='лист_тест.png'); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
rnd = random.Random(a.seed); S = 2048; sheet = Image.new('RGB', (S, S), (255, 255, 255))
def hill(w, h, tone):
    k = 2; im = Image.new('RGBA', (w * k, h * k), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pk = rnd.uniform(0.35, 0.65); dome = 0.62 * w * 0.55
    def contour(inset):
        pts = []
        for i in range(0, 121):
            t = i / 120; x = (0.04 + 0.92 * t) * w * k
            u = (t - pk) / (pk if t < pk else 1 - pk)              # -1..1, 0 в вершине
            y = (0.06 * h + dome * abs(u) ** 2.2) * k               # вершина вверху, края ниже
            pts.append((x, y))
        return pts
    top = [(0.03 * w * k, h * k)] + [(x, y) for x, y in contour(0)] + [(0.97 * w * k, h * k)]
    base = tuple(int(c * tone) for c in (74, 118, 140))
    d.polygon(top, fill=base + (255,))
    # светлые полоски под тёмной дугой и тёмный контур
    for off, col, wd in ((26, (240, 246, 244, 255), 5), (44, (170, 205, 200, 255), 4)):
        d.line([(x, y + off * k * 0.5) for x, y in contour(0)][5:-5], fill=col, width=wd * k, joint='curve')
    d.line(contour(0), fill=(24, 62, 86, 255), width=15 * k, joint='curve')
    im = im.resize((w, h), Image.LANCZOS)
    arr = np.array(im).astype(float); n = np.random.default_rng(rnd.randint(0, 9999)).normal(0, 6, (h, w)); arr[..., :3] += n[..., None] * (arr[..., 3:4] > 0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA')
for i in range(9):
    w = rnd.randint(520, 640); h = rnd.randint(520, 600); im = hill(w, h, rnd.uniform(0.9, 1.1))
    cx, cy = 340 + (i % 3) * 680, 340 + (i // 3) * 680
    sheet.paste(im, (cx - w // 2, cy - h // 2), im)
sheet.save(a.out)
