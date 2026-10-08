# Берёт плитку 2×2 (паттерн без фона) и вырезает из неё ЦЕЛЫЕ элементы: цветок + его стебель + листья на нём,
# а также ветки с листьями без цветов. Плитка повторяет каждый элемент по 4 раза; из четырёх копий берётся та,
# что дальше всего от краёв плитки (значит, целая, с продолжением стебля через бывший шов).
#
#   python элементы_из_2x2.py ситец_плитка_2x2_без-фона.png --out элементы
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('--out', default='элементы')
ap.add_argument('--radius', type=int, default=560)       # расстояние вдоль лозы от цветка, до которого листья считаются его, px плитки 2×2
ap.add_argument('--leaf-area', type=int, default=170000) # целевая площадь «ветки с листьями без цветов», px²
ap.add_argument('--pad', type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

img = Image.open(a.src).convert('RGBA')
arr = np.array(img)
FH, FW = arr.shape[:2]
T = FW // 2                                              # сторона одной плитки
S = 2
sub = arr[::S, ::S]
H, W = sub.shape[:2]
t = T // S
al = sub[..., 3]
base = al > 40
R_, G_, B_ = (sub[..., i].astype(int) for i in range(3))
pink = base & (R_ - G_ > 28) & (R_ > 140)
blue = base & (B_ - R_ > 30) & (B_ > 120)


def regions(mask, grow, minarea):
    m = ndi.binary_fill_holes(ndi.binary_dilation(mask, iterations=grow)) & base
    lab, n = ndi.label(m)
    out = []
    for k, s in enumerate(ndi.find_objects(lab), 1):
        mm = lab[s] == k
        ar = int(mm.sum())
        if ar >= minarea:
            ys, xs = np.nonzero(mm)
            out.append(dict(k=k, lab=lab, area=ar, cx=float(xs.mean() + s[1].start), cy=float(ys.mean() + s[0].start)))
    return out


flowers_all = regions(pink, 9, 2200) + regions(blue, 4, 4500)
small_blue = [r for r in regions(blue, 3, 300) if r['area'] < 4500]


def pick_canonical(items, key=lambda it: (it['cx'], it['cy']), tol=45):
    """Копии одного элемента лежат в 2×2 со сдвигом на плитку (t). Кластеризуем по положению внутри плитки
    (по модулю t, с допуском tol — центры у обрезанных краем копий плавают) и берём из кластера копию,
    ближайшую к центру 2×2 (она дальше всех от краёв, значит целая)."""
    def tdist(a_, b_):
        dx = abs(key(a_)[0] - key(b_)[0]) % t; dy = abs(key(a_)[1] - key(b_)[1]) % t
        return max(min(dx, t - dx), min(dy, t - dy)) if False else (min(dx, t - dx) ** 2 + min(dy, t - dy) ** 2) ** 0.5
    pool = sorted(items, key=lambda it: (key(it)[0] - W / 2) ** 2 + (key(it)[1] - H / 2) ** 2)   # сначала самые центральные
    chosen = []
    for it in pool:
        if all(tdist(it, c) > tol for c in chosen):
            chosen.append(it)
    return chosen


flowers = pick_canonical(flowers_all)
print(f'цветков в 2×2: {len(flowers_all)}; уникальных: {len(flowers)}')

small_mask = np.zeros_like(base)
for r in small_blue:
    small_mask |= (r['lab'] == r['k'])
core = base & ~small_mask

markers = np.zeros((H, W), np.int32)
fid = {}
for i, f in enumerate(flowers_all, 1):
    markers[(f['lab'] == f['k']) & core] = i
    fid[id(f)] = i
ws = watershed(np.zeros((H, W), np.uint8), markers, mask=core, connectivity=1)
mcp = MCP_Geometric(np.where(core, 1.0, np.inf), fully_connected=True)
dist, _ = mcp.find_costs([tuple(p) for p in np.argwhere(markers > 0)[::9]])
ws = np.where(dist <= a.radius / S, ws, 0)

green = core & (G_ - R_ > 6) & (G_ - B_ > 10)
blab, bn = ndi.label(ndi.binary_opening(green, iterations=4))
for k, s in enumerate(ndi.find_objects(blab), 1):
    sl = tuple(slice(max(0, x.start - 4), x.stop + 4) for x in s)
    m = blab[sl] == k
    vals = ws[sl][ndi.binary_dilation(m, iterations=2) & core[sl]]
    vals = vals[vals > 0]
    if len(vals):
        ws[sl][m] = np.bincount(vals).argmax()

units = []
for f in flowers:
    units.append(['цветок', ws == fid[id(f)], (f['cy'], f['cx'])])

# --- ветки с листьями без цветов: остаток лозы, нарезанный на куски нужной площади; копии убираются так же
rest = ndi.binary_opening(core & (ws == 0), iterations=2)
rlab, rn = ndi.label(rest)
target = a.leaf_area / (S * S)
rng = np.random.default_rng(1)
leaf_cands = []
for k, s in enumerate(ndi.find_objects(rlab), 1):
    m = rlab[s] == k
    ar = int(m.sum())
    if ar < 0.5 * target:
        continue
    ns = max(1, int(round(ar / target)))
    pts = np.argwhere(m)
    sel = pts[rng.choice(len(pts), size=ns, replace=False)]
    mk = np.zeros(m.shape, np.int32)
    for j, (y, x) in enumerate(sel, 1):
        mk[y, x] = j
    w2 = watershed(np.zeros(m.shape, np.uint8), mk, mask=m, connectivity=1)
    for j in range(1, ns + 1):
        mm = w2 == j
        if mm.sum() < 0.4 * target:
            continue
        ys, xs = np.nonzero(mm)
        full = np.zeros((H, W), bool)
        full[s][mm] = True
        leaf_cands.append(dict(cx=float(xs.mean() + s[1].start), cy=float(ys.mean() + s[0].start), mask=full))
leafs = pick_canonical(leaf_cands)
for l in leafs:
    units.append(['ветка', l['mask'], (l['cy'], l['cx'])])

# --- незабудки: кластеры
cl = ndi.binary_dilation(small_mask, iterations=14)
clab, cn = ndi.label(cl)
sm = []
for k, s in enumerate(ndi.find_objects(clab), 1):
    m = clab[s] == k
    nfl = len(np.unique(ndi.label(small_mask[s] & m)[0])) - 1
    ys, xs = np.nonzero(m)
    full = np.zeros((H, W), bool)
    full[s][m] = True
    sm.append(dict(cx=float(xs.mean() + s[1].start), cy=float(ys.mean() + s[0].start), mask=full, n=nfl))
sm = [c for c in pick_canonical(sm) if c['n'] >= 2]
sm.sort(key=lambda c: -c['n'])
for c in sm[:4]:
    mask = (c['mask'] & base) | (ndi.binary_dilation(c['mask'] & small_mask, iterations=3) & base)
    units.append(['незабудки', mask, (c['cy'], c['cx'])])

order = {'цветок': 0, 'ветка': 1, 'незабудки': 2}
units.sort(key=lambda u: (order[u[0]], u[2]))
print(f'итого: {sum(u[0]=="цветок" for u in units)} цветков, {sum(u[0]=="ветка" for u in units)} веток, {sum(u[0]=="незабудки" for u in units)} групп незабудок')


# --- вывод
def to_white(mh):
    ys, xs = np.nonzero(mh)
    x0, y0 = max(0, xs.min() - a.pad // S) * S, max(0, ys.min() - a.pad // S) * S
    x1, y1 = min(FW, (xs.max() + a.pad // S + 1) * S), min(FH, (ys.max() + a.pad // S + 1) * S)
    ms = ndi.binary_dilation(np.kron(mh[y0 // S:y1 // S, x0 // S:x1 // S], np.ones((S, S), bool)), iterations=2)
    cut = arr[y0:y0 + ms.shape[0], x0:x0 + ms.shape[1]]
    ms = ms[:cut.shape[0], :cut.shape[1]]
    alpha = (cut[..., 3].astype(float) / 255.0) * ms
    rgb = cut[..., :3].astype(float) * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    return Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(np.dstack([cut[..., :3], (alpha * 255).astype(np.uint8)]), 'RGBA')


cell, cols = 380, 6
rows = -(-len(units) // cols)
sheet = Image.new('RGB', (cols * cell, rows * cell), (255, 255, 255))
d = ImageDraw.Draw(sheet)
fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
meta, cnt = [], {}
over = np.array(img.convert('RGB').resize((W, H))).copy()
rng2 = np.random.default_rng(7)
palette = [tuple(int(v) for v in rng2.integers(40, 255, 3)) for _ in range(len(units))]
for i, (kind, m, _) in enumerate(units, 1):
    cnt[kind] = cnt.get(kind, 0) + 1
    nm = f'{i:02d}_{kind}_{cnt[kind]}'
    white, rgba = to_white(m)
    white.save(os.path.join(a.out, nm + '.png'))
    rgba.save(os.path.join(a.out, nm + '_прозр.png'))
    th = white.copy(); th.thumbnail((cell - 12, cell - 44))
    ox, oy = (i - 1) % cols * cell, (i - 1) // cols * cell
    sheet.paste(th, (ox + 6, oy + 38)); d.text((ox + 8, oy + 4), f'{i} {kind}', fill=(0, 0, 0), font=fnt)
    e = m & ~ndi.binary_erosion(m, iterations=3)
    over[e] = palette[i - 1]
    meta.append(dict(n=i, kind=kind, name=nm, size=white.size))
sheet.save(os.path.join(a.out, '_сводка.jpg'), quality=92)
ov = Image.fromarray(over); do = ImageDraw.Draw(ov)
fo = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
for i, (kind, m, (cy, cx)) in enumerate(units, 1):
    do.text((cx - 14, cy - 20), str(i), fill=(255, 255, 255), font=fo, stroke_width=4, stroke_fill=palette[i - 1])
ov.save(os.path.join(a.out, '_разметка_на_плитке.jpg'), quality=90)
json.dump(meta, open(os.path.join(a.out, '_список.json'), 'w'), ensure_ascii=False, indent=1)
for mm in meta:
    print(f"  {mm['n']:2d} {mm['kind']:10s} {mm['size'][0]}×{mm['size'][1]}")
