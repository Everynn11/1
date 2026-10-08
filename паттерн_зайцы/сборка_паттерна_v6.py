# Сборка паттерна «заяц на цветущем лугу»: однотонный фон + элементы, бесшовно (тор).
# Основа — сборка_паттерна_v5.py (утки). Отличия от v5:
#  - фон однотонный (Linen), элементы берутся по префиксу имени файла (см. CM);
#  - размер элемента задаётся в сантиметрах на ткани (калибр = корень из ширина×высота габарита),
#    так что высокие узкие и низкие широкие элементы выходят соразмерными;
#  - одинаковые позы зайцев не ставятся рядом (--hare-gap);
#  - часть бабочек «садится» на кончик колоса или на цветок (--perch).
# Раппорт 29" = 4350 px @150 dpi, высота 5200 px.
import argparse, glob, math, os, random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PX_CM = 150 / 2.54

# калибр элемента на ткани, см (правьте здесь)
CM = {'заяц': 9.5, 'малиновка': 6.5, 'бабочка': 4.5, 'анемона': 7.0, 'бутон': 6.0,
      'папоротник': 10.0, 'эвкалипт': 9.0, 'ягоды': 9.0, 'дуб': 10.0, 'колос': 7.5, 'филлер': 6.0}

# (метка, префиксы, аргумент-количество, отступ между элементами, см)
GROUPS = [('ветки',   ('папоротник', 'эвкалипт', 'ягоды', 'дуб'), 'branches',    2.0),
          ('зайцы',   ('заяц',),                                  'hares',       1.2),
          ('птицы',   ('малиновка',),                             'birds',       1.0),
          ('цветы',   ('анемона', 'бутон'),                       'flowers',     0.9),
          ('колосья', ('колос',),                                 'spikes',      0.7),
          ('бабочки', ('бабочка',),                               'butterflies', 0.5),
          ('филлеры', ('филлер',),                                'fillers',     0.4)]

ap = argparse.ArgumentParser()
ap.add_argument('--shelf', required=True)            # папка с вырезанными PNG (имена по префиксам)
ap.add_argument('--out', default='раппорт.jpg')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--width', type=int, default=4350)
ap.add_argument('--height', type=int, default=5200)
ap.add_argument('--scale', type=float, default=1.0)  # 0.25 = черновик
ap.add_argument('--bg', default='ECE5D8')            # Linen
ap.add_argument('--quality', type=int, default=95)
ap.add_argument('--hares', type=int, default=10)
ap.add_argument('--branches', type=int, default=12)
ap.add_argument('--flowers', type=int, default=16)
ap.add_argument('--birds', type=int, default=4)
ap.add_argument('--butterflies', type=int, default=7)
ap.add_argument('--spikes', type=int, default=8)
ap.add_argument('--fillers', type=int, default=16)
ap.add_argument('--perch', type=int, default=3)      # сколько бабочек сажать на колос/цветок
ap.add_argument('--hare-gap', type=float, default=0.35)  # мин. расстояние между одинаковыми позами, доля ширины
ap.add_argument('--check', action='store_true')      # сохранить плитку 2×2 для проверки швов
ap.add_argument('--list', action='store_true')       # показать найденные элементы и выйти
a = ap.parse_args()

rnd = random.Random(a.seed)
S = a.scale
W, H = int(a.width * S), int(a.height * S)
bgc = tuple(int(a.bg[i:i+2], 16) for i in (0, 2, 4))
files = sorted(glob.glob(os.path.join(a.shelf, '*.png')))


def prefix_of(name):
    return next((p for p in CM if name.startswith(p)), None)


def load_group(prefixes):
    out = []
    for f in files:
        name = os.path.basename(f)
        if prefix_of(name) in prefixes:
            im = Image.open(f).convert('RGBA')
            bb = im.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox()
            out.append((name, im.crop(bb), CM[prefix_of(name)]))
    return out


loaded = {lab: load_group(pf) for lab, pf, _, _ in GROUPS}
unmatched = [os.path.basename(f) for f in files if prefix_of(os.path.basename(f)) is None]
if unmatched:
    print('НЕ ОПОЗНАНЫ (имя должно начинаться с префикса):', *unmatched, sep='\n  ')
if a.list:
    for lab, g in loaded.items():
        print(f'{lab}: {len(g)}', [n for n, _, _ in g])
    raise SystemExit

canvas = Image.new('RGB', (W, H), bgc)
D = 10                                  # масштаб маски занятости
MW, MH = -(-W // D), -(-H // D)
occ = np.zeros((MH, MW), bool)
hare_log = []                           # (поза, cx, cy)
anchors = []                            # (x, y) кончиков колосьев и цветков, куда садятся бабочки
warn_up = {}                            # имя -> во сколько раз растянут исходник


def prep(img, name, cm, cm_scale=1.0):
    w0, h0 = img.size
    k = cm * cm_scale * PX_CM * S * rnd.uniform(0.88, 1.12) / math.sqrt(w0 * h0)
    if k / S > 1.3:
        warn_up[name] = max(warn_up.get(name, 0), round(k / S, 2))
    el = img.resize((max(1, int(w0 * k)), max(1, int(h0 * k))), Image.LANCZOS)
    if rnd.random() < 0.5:
        el = el.transpose(Image.FLIP_LEFT_RIGHT)
    return el


def mask_idx(el, margin_px):
    w, h = el.size
    al = np.array(el.getchannel('A')) > 40
    m = np.array(Image.fromarray((al * 255).astype(np.uint8)).resize(
        (max(1, w // D), max(1, h // D)), Image.BILINEAR)) > 40
    m = ndi.binary_dilation(m, structure=ndi.generate_binary_structure(2, 1),
                            iterations=max(1, int(margin_px / D)))
    return np.nonzero(m)


def tor_dist(x0, y0, x1, y1):
    dx, dy = abs(x0 - x1), abs(y0 - y1)
    return math.hypot(min(dx, W - dx), min(dy, H - dy))


def mark(el, x, y, margin_px):
    ys, xs = mask_idx(el, margin_px)
    occ[(ys + y // D) % MH, (xs + x // D) % MW] = True


def place_free(el, margin_px, pose=None, tries=600):
    w, h = el.size
    ys, xs = mask_idx(el, margin_px)
    for _ in range(tries):
        cx, cy = rnd.randrange(W), rnd.randrange(H)
        if pose and any(p == pose and tor_dist(cx, cy, x, y) < a.hare_gap * W for p, x, y in hare_log):
            continue
        x, y = cx - w // 2, cy - h // 2
        yy, xx = (ys + y // D) % MH, (xs + x // D) % MW
        if not occ[yy, xx].any():
            occ[yy, xx] = True
            if pose:
                hare_log.append((pose, cx, cy))
            return x, y
    return None


def top_anchor(el):
    al = np.array(el.getchannel('A')) > 40
    rows = np.nonzero(al.any(axis=1))[0]
    r0, r1 = rows[0], rows[0] + max(1, int(0.15 * (rows[-1] - rows[0])))
    xs = np.nonzero(al[r0:r1].any(axis=0))[0]
    return int(xs.mean()), int(r0 + 0.06 * el.height)


def paste(el, x, y):
    w, h = el.size
    rgb, al = el.convert('RGB'), el.getchannel('A')
    for dx in (-W, 0, W):               # вылезшее за край возвращается с противоположной стороны
        for dy in (-H, 0, H):
            xx, yy = x + dx, y + dy
            if xx < W and yy < H and xx + w > 0 and yy + h > 0:
                canvas.paste(rgb, (xx, yy), al)


placed = {}
for lab, prefixes, arg, margin_cm in GROUPS:
    group, n = loaded[lab], getattr(a, arg)
    if not group:
        if n: print(f'нет элементов «{lab}», пропуск')
        continue
    placed[lab] = 0
    margin_px = margin_cm * PX_CM * S
    order = []                          # позы идут по кругу из перемешанного списка
    for i in range(n):
        if not order:
            order = list(range(len(group))); rnd.shuffle(order)
        name, img, cm = group[order.pop()]
        if lab == 'бабочки' and i < min(a.perch, len(anchors)):
            ax, ay = anchors[rnd.randrange(len(anchors))]   # садится на кончик колоса или цветок
            el = prep(img, name, cm, 0.9)
            x = ax + rnd.choice((-1, 1)) * int(0.25 * el.width) - el.width // 2
            y = ay - el.height // 2
            mark(el, x, y, margin_px); paste(el, x, y); placed[lab] += 1
            continue
        el = prep(img, name, cm)
        pos = place_free(el, margin_px, pose=name if lab == 'зайцы' else None)
        if pos:
            x, y = pos
            paste(el, x, y); placed[lab] += 1
            if prefix_of(name) in ('колос', 'анемона'):
                tx, ty = top_anchor(el)
                anchors.append((x + tx, y + ty))

print({k: f'{v} из {getattr(a, next(g[2] for g in GROUPS if g[0] == k))}' for k, v in placed.items()})
if warn_up:
    print('Исходник растянут больше чем в 1,3×, края могут быть мягкими:', warn_up)
canvas.save(a.out, quality=a.quality, subsampling=0, optimize=True)
print('готово', a.out, canvas.size)
if a.check:
    t = canvas.resize((max(1, W // 6), max(1, H // 6)), Image.LANCZOS)
    c = Image.new('RGB', (t.width * 2, t.height * 2))
    for x in (0, 1):
        for y in (0, 1):
            c.paste(t, (x * t.width, y * t.height))
    chk = os.path.splitext(a.out)[0] + '_плитка2x2.jpg'
    c.save(chk, quality=88)
    print('плитка для проверки швов:', chk)
