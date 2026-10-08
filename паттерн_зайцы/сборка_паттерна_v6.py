# Сборка паттерна «заяц на цветущем лугу»: однотонный фон + элементы, бесшовно (тор).
# Основа — сборка_паттерна_v5.py (утки). Отличия: фон однотонный, элементы берутся
# по префиксу имени файла, одинаковые позы зайцев не ставятся рядом.
import argparse, glob, os, random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--shelf', required=True)          # папка с вырезанными PNG
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--width', type=int, default=4350)   # раппорт 29" @150 dpi
ap.add_argument('--height', type=int, default=5200)
ap.add_argument('--scale', type=float, default=1.0)  # 0.25 = черновик
ap.add_argument('--bg', default='ECE5D8')            # Linen
ap.add_argument('--quality', type=int, default=95)
ap.add_argument('--hares', type=int, default=10)
ap.add_argument('--branches', type=int, default=12)  # папоротник/эвкалипт/ягоды/дуб
ap.add_argument('--flowers', type=int, default=16)   # анемоны и бутоны
ap.add_argument('--birds', type=int, default=4)
ap.add_argument('--butterflies', type=int, default=7)
ap.add_argument('--spikes', type=int, default=8)
ap.add_argument('--fillers', type=int, default=16)
ap.add_argument('--hare-gap', type=float, default=0.35)  # мин. расстояние между одинаковыми позами, доля ширины
a = ap.parse_args()

rnd = random.Random(a.seed)
S = a.scale
W, H = int(a.width * S), int(a.height * S)
bgc = tuple(int(a.bg[i:i+2], 16) for i in (0, 2, 4))


def load_group(*prefixes):
    files = sorted(f for p in prefixes for f in glob.glob(os.path.join(a.shelf, p + '*.png')))
    out = []
    for f in files:
        im = Image.open(f).convert('RGBA')
        bb = im.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox()
        out.append((os.path.basename(f), im.crop(bb)))
    return out


canvas = Image.new('RGB', (W, H), bgc)
D = 10                                  # масштаб маски занятости
MW, MH = W // D + 1, H // D + 1
occ = np.zeros((MH, MW), bool)
hare_log = []                           # (поза, cx, cy)


def tor_dist(x0, y0, x1, y1):
    dx = abs(x0 - x1); dy = abs(y0 - y1)
    return np.hypot(min(dx, W - dx), min(dy, H - dy))


def place(img, width, margin_px, pose=None, tries=600):
    k = width * rnd.uniform(0.88, 1.12) / img.width
    w, h = max(1, int(img.width * k)), max(1, int(img.height * k))
    el = img.resize((w, h), Image.LANCZOS)
    if rnd.random() < 0.5:
        el = el.transpose(Image.FLIP_LEFT_RIGHT)
    al = np.array(el.getchannel('A')) > 40
    m = np.array(Image.fromarray((al * 255).astype(np.uint8)).resize(
        (max(1, w // D), max(1, h // D)), Image.BILINEAR)) > 40
    m = ndi.binary_dilation(m, structure=ndi.generate_binary_structure(2, 1),
                            iterations=max(1, int(margin_px / D)))
    ys, xs = np.nonzero(m)
    for _ in range(tries):
        cx, cy = rnd.randrange(W), rnd.randrange(H)
        if pose is not None and any(p == pose and tor_dist(cx, cy, x, y) < a.hare_gap * W
                                    for p, x, y in hare_log):
            continue
        x0, y0 = (cx - w // 2) // D, (cy - h // 2) // D
        yy = (ys + y0) % MH; xx = (xs + x0) % MW
        if not occ[yy, xx].any():
            occ[yy, xx] = True
            if pose is not None:
                hare_log.append((pose, cx, cy))
            return el, cx - w // 2, cy - h // 2
    return None


def paste(el, x, y):
    w, h = el.size
    rgb = el.convert('RGB'); al = el.getchannel('A')
    for dx in (-W, 0, W):               # всё, что вылезает за край, возвращается с другой стороны
        for dy in (-H, 0, H):
            xx, yy = x + dx, y + dy
            if xx < W and yy < H and xx + w > 0 and yy + h > 0:
                canvas.paste(rgb, (xx, yy), al)


placed = {}


def run(kind, group, n, width, margin, hare=False):
    if not group:
        if n: print(f'нет файлов для «{kind}», пропуск')
        return
    placed[kind] = 0
    order = []                          # поза берётся по кругу из перемешанного списка
    for i in range(n):
        if not order:
            order = list(range(len(group))); rnd.shuffle(order)
        j = order.pop()
        name, img = group[j]
        r = place(img, width * S, margin * S, pose=name if hare else None)
        if r:
            paste(*r); placed[kind] += 1


# масштабы на раппорте 29" (1 см ≈ 59 px): ветка ~17 см, заяц ~10 см, цветок ~7 см
run('ветки',    load_group('папоротник', 'эвкалипт', 'ягоды', 'дуб'), a.branches, 1000, 60)
run('зайцы',    load_group('заяц'),                                  a.hares,    600, 70, hare=True)
run('цветы',    load_group('анемона', 'бутон'),                      a.flowers,  420, 50)
run('птицы',    load_group('малиновка'),                             a.birds,    360, 60)
run('бабочки',  load_group('бабочка'),                               a.butterflies, 260, 50)
run('колосья',  load_group('колос'),                                 a.spikes,   520, 40)
run('филлеры',  load_group('филлер'),                                a.fillers,  300, 30)
print(placed)
canvas.save(a.out, quality=a.quality, subsampling=0, optimize=True)
print('готово', a.out, canvas.size)
