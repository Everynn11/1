# Режет бесшовный паттерн без фона на ЦЕЛЫЕ веточки: цветок + его куски лозы и листья, и ветки с листьями без цветов.
# Паттерн бесшовный, поэтому части одного элемента лежат по разные стороны шва (стебель внизу картинки, бутон вверху).
# Чтобы собрать их вместе, работаем на плитке 3×3 (край картинки там непрерывен) и для каждого элемента берём одну копию
# (ту, чей центр лежит в центральной плитке). Заготовки грубые: на швах возможны стыки — Gemini перерисует начисто.
#
#   python разделить_на_элементы.py ситец_1_без-фона.png --out элементы
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('--out', default='элементы')
ap.add_argument('--radius', type=int, default=420)       # макс. расстояние по лозе от цветка до его листьев, px исходника
ap.add_argument('--leaf-area', type=int, default=110000) # целевая площадь ветки с листьями без цветов, px исходника²
ap.add_argument('--pad', type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

img = Image.open(a.src).convert('RGBA')
arr0 = np.array(img)
h0, w0 = arr0.shape[:2]
arr = np.tile(arr0, (3, 3, 1))                          # плитка 3×3 (полное разрешение, для вырезки)
S = 2                                                    # анализ в половинном разрешении
sub = arr[::S, ::S]
H, W = sub.shape[:2]
th, tw = h0 // S, w0 // S                                # размер одной плитки в анализе
inside = lambda cx, cy: tw <= cx < 2 * tw and th <= cy < 2 * th

al = sub[..., 3]
base = al > 40
R, G, B = (sub[..., i].astype(int) for i in range(3))
pink = base & (R - G > 28) & (R > 140)
blue = base & (B - R > 30) & (B > 120)


def regions(mask, grow, minarea):
    m = ndi.binary_fill_holes(ndi.binary_dilation(mask, iterations=grow)) & base
    lab, n = ndi.label(m)
    sl = ndi.find_objects(lab)
    areas = ndi.sum(m, lab, range(1, n + 1))
    out = []
    for k, (s, ar) in enumerate(zip(sl, areas), 1):
        if ar < minarea:
            continue
        ys, xs = np.nonzero(lab[s] == k)
        out.append(dict(k=k, lab=lab, area=int(ar), cx=int(xs.mean() + s[1].start), cy=int(ys.mean() + s[0].start),
                        bb=(s[1].start, s[0].start, s[1].stop, s[0].stop)))
    return out


pk = regions(pink, 9, 2200)
bl_big = regions(blue, 4, 4500)
flowers_all = pk + bl_big
flowers_in = [f for f in flowers_all if inside(f['cx'], f['cy'])]
print(f'цветков во всей плитке {len(flowers_all)}, уникальных (центр в центральной плитке) {len(flowers_in)}')

small_blue = regions(blue, 3, 300)
small_blue = [r for r in small_blue if r['area'] < 4500]
small_mask = np.zeros_like(base)
for r in small_blue:
    small_mask |= (r['lab'] == r['k'])
core = base & ~small_mask

# --- ближайший по лозе цветок
markers = np.zeros((H, W), np.int32)
for i, f in enumerate(flowers_all, 1):
    markers[(f['lab'] == f['k']) & core] = i
ws = watershed(np.zeros((H, W), np.uint8), markers, mask=core, connectivity=1)
mcp = MCP_Geometric(np.where(core, 1.0, np.inf), fully_connected=True)
starts = np.argwhere(markers > 0)[::9]
dist, _ = mcp.find_costs([tuple(p) for p in starts])
ws = np.where(dist <= a.radius / S, ws, 0)

# --- листья цельными: листовая «капля» целиком достаётся цветку, у которого больше её пикселей
green = core & (G - R > 6) & (G - B > 10)
blab, bn = ndi.label(ndi.binary_opening(green, iterations=4))
for k, s in enumerate(ndi.find_objects(blab), 1):
    sl = tuple(slice(max(0, x.start - 4), x.stop + 4) for x in s)
    m = (blab[sl] == k)
    vals = ws[sl][ndi.binary_dilation(m, iterations=2) & core[sl]]
    vals = vals[vals > 0]
    if len(vals):
        ws[sl][m] = np.bincount(vals).argmax()

units = []                                               # (вид, маска в координатах плитки, ключ для сортировки)
for i, f in enumerate(flowers_all, 1):
    if inside(f['cx'], f['cy']):
        m = ws == i
        ys, xs = np.nonzero(m)
        units.append(('цветок', m, (f['cy'], f['cx'])))

# --- остатки лозы далеко от цветов: ветки с листьями без цветов (порезаны на куски нужного размера)
rest = ndi.binary_opening(core & (ws == 0), iterations=2)
rlab, rn = ndi.label(rest)
target = a.leaf_area / (S * S)
rng = np.random.default_rng(1)
leaf_units = []
for k, s in enumerate(ndi.find_objects(rlab), 1):
    m = rlab[s] == k
    ar = int(m.sum())
    if ar < 0.5 * target:
        continue
    if not (s[1].stop > tw and s[1].start < 2 * tw and s[0].stop > th and s[0].start < 2 * th):
        continue                                          # не пересекает центральную плитку
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
        cy, cx = ys.mean() + s[0].start, xs.mean() + s[1].start
        if inside(cx, cy):
            full = np.zeros((H, W), bool)
            full[s][mm] = True
            leaf_units.append(('ветка', full, (cy, cx)))
units += leaf_units

# --- незабудки: кластеры мелких синих цветков
cl = ndi.binary_dilation(small_mask, iterations=14)
clab, cn = ndi.label(cl)
sm_units = []
for k, s in enumerate(ndi.find_objects(clab), 1):
    m = clab[s] == k
    nflow = len(np.unique(ndi.label(small_mask[s] & m)[0])) - 1
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() + s[0].start, xs.mean() + s[1].start
    if inside(cx, cy):
        full = np.zeros((H, W), bool)
        full[s][m] = True
        sm_units.append((nflow, ('незабудки', full & base | ndi.binary_dilation(full & small_mask, iterations=3) & base, (cy, cx))))
sm_units.sort(key=lambda t: -t[0])
chosen = [u for n, u in sm_units if n >= 2][:3] + [u for n, u in sm_units if n == 1][:2]
units += chosen
print(f'элементов: {sum(1 for u in units if u[0]=="цветок")} цветков, {len(leaf_units)} веток с листьями, {len(chosen)} групп незабудок')

units.sort(key=lambda u: ({'цветок': 0, 'ветка': 1, 'незабудки': 2}[u[0]], u[2]))


# --- вывод: каждая заготовка на белом фоне (и отдельно с прозрачностью)
def to_white(m_half):
    ys, xs = np.nonzero(m_half)
    x0, y0 = max(0, (xs.min() - a.pad // S)) * S, max(0, (ys.min() - a.pad // S)) * S
    x1, y1 = min(arr.shape[1], (xs.max() + a.pad // S + 1) * S), min(arr.shape[0], (ys.max() + a.pad // S + 1) * S)
    msub = np.kron(m_half[y0 // S:y1 // S, x0 // S:x1 // S], np.ones((S, S), bool))
    msub = ndi.binary_dilation(msub, iterations=2)       # половинное разрешение теряет край, возвращаем
    cut = arr[y0:y0 + msub.shape[0], x0:x0 + msub.shape[1]]
    msub = msub[:cut.shape[0], :cut.shape[1]]
    alpha = (cut[..., 3].astype(float) / 255.0) * msub
    rgb = cut[..., :3].astype(float) * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    return Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(np.dstack([cut[..., :3], (alpha * 255).astype(np.uint8)]), 'RGBA')


cell, cols = 360, 6
rows = -(-len(units) // cols)
sheet = Image.new('RGB', (cols * cell, rows * cell), (255, 255, 255))
d = ImageDraw.Draw(sheet)
fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
meta, counter = [], {}
for i, (kind, m, _) in enumerate(units, 1):
    counter[kind] = counter.get(kind, 0) + 1
    nm = f'{i:02d}_{kind}_{counter[kind]}'
    white, rgba = to_white(m)
    white.save(os.path.join(a.out, nm + '.png'))
    rgba.save(os.path.join(a.out, nm + '_прозр.png'))
    t = white.copy()
    t.thumbnail((cell - 12, cell - 44))
    ox, oy = (i - 1) % cols * cell, (i - 1) // cols * cell
    sheet.paste(t, (ox + 6, oy + 38))
    d.text((ox + 8, oy + 4), f'{i} {kind}', fill=(0, 0, 0), font=fnt)
    meta.append(dict(n=i, kind=kind, name=nm, size=white.size))
sheet.save(os.path.join(a.out, '_сводка.jpg'), quality=92)
json.dump(meta, open(os.path.join(a.out, '_список.json'), 'w'), ensure_ascii=False, indent=1)
print(f'готово: {len(units)} заготовок → {a.out}')
for mm in meta:
    print(f"  {mm['n']:2d} {mm['kind']:10s} {mm['size'][0]}×{mm['size'][1]}")
