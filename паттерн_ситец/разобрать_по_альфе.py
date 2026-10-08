# Разбирает паттерн без фона на отдельные части по альфа-каналу (связные куски).
# Куски, обрезанные краем картинки, достраиваются: ищется продолжение на противоположном краю
# (по горизонтали и по вертикали, паттерн бесшовный) и «докидывается» с нужным сдвигом.
# Получаются цельные заготовки на белом фоне; Gemini перерисует их начисто.
#
#   python разобрать_по_альфе.py ситец_1_без-фона.png --out части
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('--out', default='части')
ap.add_argument('--tol', type=int, default=36)          # допуск совпадения места пересечения шва, px (швы в паттерне «плавают»)
ap.add_argument('--min-rows', type=int, default=6)      # сколько строк/столбцов должны совпасть, чтобы считать куски продолжением друг друга
ap.add_argument('--giant', type=int, default=300000)    # куски больше этой площади — «лозы» (к ним докидываем остальное)
ap.add_argument('--pad', type=int, default=24)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

arr = np.array(Image.open(a.src).convert('RGBA'))
H, W = arr.shape[:2]
al = arr[..., 3] > 40
lab, n = ndi.label(al, structure=np.ones((3, 3)))
masks = {}
for k, s in enumerate(ndi.find_objects(lab), 1):
    m = lab[s] == k
    if m.sum() >= 120:                                   # крошки не берём
        masks[k] = (s, m)
area = {k: int(m.sum()) for k, (s, m) in masks.items()}


def edge_profile(k, side):
    s, m = masks[k]
    full = np.zeros((H, W), bool)
    full[s][m] = True
    if side == 'L': v = full[:, 0:2].any(axis=1)
    elif side == 'R': v = full[:, -2:].any(axis=1)
    elif side == 'T': v = full[0:2, :].any(axis=0)
    else: v = full[-2:, :].any(axis=0)
    return v


edges = {k: {sd: edge_profile(k, sd) for sd in 'LRTB'} for k in masks}
prof_d = {k: {sd: ndi.binary_dilation(p, iterations=a.tol) for sd, p in e.items()} for k, e in edges.items()}

# пары: (кусок A, кусок B, сдвиг B относительно A, оценка)
pairs = []
for ka in masks:
    for kb in masks:
        if ka == kb:
            continue
        # B лежит справа от A через правый шов: A.R ↔ B.L ; сдвиг B = (+W, 0)
        sc = int((prof_d[ka]['R'] & edges[kb]['L']).sum())
        if sc >= a.min_rows: pairs.append((ka, kb, (W, 0), sc))
        # B лежит под A через нижний шов: A.B ↔ B.T ; сдвиг (0, +H)
        sc = int((prof_d[ka]['B'] & edges[kb]['T']).sum())
        if sc >= a.min_rows: pairs.append((ka, kb, (0, H), sc))

# --- кто к кому докидывается: лозы остаются якорями, остальное «приклеивается» к самому подходящему соседу
giants = [k for k in masks if area[k] >= a.giant]
parent = {}                                              # k -> (родитель, сдвиг k относительно родителя)
for k in masks:
    if k in giants:
        continue
    cands = []
    for ka, kb, sh, sc in pairs:
        if kb == k and ka != k:                          # k лежит правее/ниже ka
            cands.append((sc, ka, sh))
        if ka == k and kb != k:                          # k лежит левее/выше kb → сдвиг k = -sh относительно kb
            cands.append((sc, kb, (-sh[0], -sh[1])))
    if cands:
        cands.sort(key=lambda c: (-(c[1] in giants), -c[0]))   # сначала лозы, потом по оценке
        sc, par, sh = cands[0]
        parent[k] = (par, sh, sc)

# цепочки: поднимаемся до якоря (лозы или самостоятельного куска), суммируя сдвиги
def root_of(k, seen=()):
    if k in giants or k not in parent or k in seen:
        return k, (0, 0)
    par, sh, _ = parent[k]
    r, off = root_of(par, seen + (k,))
    return r, (off[0] + sh[0], off[1] + sh[1])


groups = {}
for k in masks:
    r, off = root_of(k)
    groups.setdefault(r, []).append((k, off))

# --- вывод
fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
report, sheet_items = [], []
big = sorted(groups.items(), key=lambda g: -sum(area[k] for k, _ in g[1]))
for gi, (r, members) in enumerate(big, 1):
    xs0 = min(off[0] for _, off in members); ys0 = min(off[1] for _, off in members)
    x1 = max(off[0] + W for _, off in members if True)
    # реальные границы по bbox каждого куска
    bx0 = min(off[0] + masks[k][0][1].start for k, off in members); bx1 = max(off[0] + masks[k][0][1].stop for k, off in members)
    by0 = min(off[1] + masks[k][0][0].start for k, off in members); by1 = max(off[1] + masks[k][0][0].stop for k, off in members)
    bx0 -= a.pad; by0 -= a.pad; bx1 += a.pad; by1 += a.pad
    cw, ch = bx1 - bx0, by1 - by0
    canvas = np.zeros((ch, cw, 4), np.uint8)
    for k, off in sorted(members, key=lambda t: -area[t[0]]):
        s, m = masks[k]
        y0, x0 = s[0].start + off[1] - by0, s[1].start + off[0] - bx0
        sub = arr[s][..., :4].copy()
        sub[..., 3] = np.where(m, sub[..., 3], 0)
        region = canvas[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
        top = Image.fromarray(sub)
        base_im = Image.fromarray(region.copy())
        base_im.alpha_composite(top)
        canvas[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = np.array(base_im)
    rgba = Image.fromarray(canvas, 'RGBA')
    white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    kinds = 'лоза' if r in giants else ('цветок/кусок' if sum(area[k] for k, _ in members) > 20000 else 'мелочь')
    nm = f'{gi:02d}_{kinds.replace("/", "-")}'
    white.convert('RGB').save(os.path.join(a.out, nm + '.png'))
    rgba.save(os.path.join(a.out, nm + '_прозр.png'))
    report.append(dict(n=gi, имя=nm, размер=[cw, ch], куски=[k for k, _ in members], докинуто=[(k, off) for k, off in members if off != (0, 0)]))
    sheet_items.append((gi, kinds, white.convert('RGB'), len(members)))

cell, cols = 380, 6
rows = -(-len(sheet_items) // cols)
sheet = Image.new('RGB', (cols * cell, rows * cell), (255, 255, 255))
d = ImageDraw.Draw(sheet)
for i, (gi, kinds, im, nm) in enumerate(sheet_items):
    t = im.copy(); t.thumbnail((cell - 12, cell - 46))
    ox, oy = i % cols * cell, i // cols * cell
    sheet.paste(t, (ox + 6, oy + 40)); d.text((ox + 8, oy + 4), f'{gi} {kinds} ({nm})', fill=(0, 0, 0), font=fnt)
sheet.save(os.path.join(a.out, '_сводка.jpg'), quality=90)
json.dump(report, open(os.path.join(a.out, '_отчёт.json'), 'w'), ensure_ascii=False, indent=1, default=int)
print(f'кусков по альфе: {len(masks)}; лоз: {len(giants)}; готовых заготовок: {len(groups)}')
for r_ in report:
    print(f"  {r_['n']:2d} {r_['имя']:22s} {r_['размер'][0]}×{r_['размер'][1]}  кусков {len(r_['куски'])}  докинуто {len(r_['докинуто'])}")
