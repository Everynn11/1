# Из дуг, снятых с образца (дуги.json), делает направляющие листы для Gemini: 3×3 силуэта холмиков
# (серая линия = дуга, светло-серая область = холмик с длинной юбкой). Кривые сглаживаются и слегка варьируются
# (±разброс по ширине/высоте/зеркало), это геометрия формы, а не копия рисунка.
#   python -I направляющие_дуги.py --arcs дуги.json --out направляющие --n 18
import argparse, json, os, random
import numpy as np
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument('--arcs', default='дуги.json'); ap.add_argument('--out', default='направляющие')
ap.add_argument('--n', type=int, default=18); ap.add_argument('--seed', type=int, default=4); ap.add_argument('--size', type=int, default=2000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
raw = json.load(open(a.arcs)); items = []
for c in raw:
    x = np.array(c['x']); y = np.array(c['y']); w = x.max() - x.min()
    u = (x - x.min()) / w; p = np.polyfit(u, y, 3); res = float(np.sqrt(np.mean((np.polyval(p, u) - y) ** 2)) / w)
    g = np.linspace(0, 1, 60); yy = np.polyval(p, g); yy = yy - yy.max()           # y вверх отрицательный: вершина = min
    rise = float(-yy.min()) / w; apex = float(g[yy.argmin()]); tilt = float((yy[-1] - yy[0]) / w)
    dy = np.diff(yy); sgn = np.sign(dy[np.abs(dy) > 1e-3 * w]); changes = int(np.sum(sgn[1:] != sgn[:-1]))
    ends = min(yy[0], yy[-1]) - yy.min()                              # оба конца ниже вершины (горб, а не S-образная кривая)
    if res > 0.05 or not (0.12 < apex < 0.88) or changes > 1 or -yy.min() < 0.07 * w or abs(yy[0] - yy.min()) < 0.03 * w or abs(yy[-1] - yy.min()) < 0.03 * w: continue
    items.append(dict(w=w, g=g, y=yy, res=res, rise=rise, apex=apex, tilt=tilt))
print('годных дуг:', len(items))
# выбор самых разных: фермерская выборка по признакам (ширина, подъём, вершина, наклон)
F = np.array([[i['w'] / 200, i['rise'] * 4, i['apex'] * 2, i['tilt'] * 4] for i in items]); pick = [int(np.argmin([i['res'] for i in items]))]
while len(pick) < min(a.n, len(items)):
    d = np.min(np.linalg.norm(F[:, None] - F[pick][None], axis=2), axis=1); d[pick] = -1; pick.append(int(d.argmax()))
sel = [items[i] for i in pick]; wmax = max(i['w'] for i in sel)
S = a.size; cell = S // 3; sheets = (len(sel) + 8) // 9; scale = 0.80 * cell / wmax      # самый широкий холмик = 80% клетки, высота с юбкой 1.05 ширины влезает
for sh in range(sheets):
    im = Image.new('RGB', (S, S), (255, 255, 255)); d = ImageDraw.Draw(im)
    for k, it in enumerate(sel[sh * 9:sh * 9 + 9]):
        r_, c_ = divmod(k, 3); cx, cy = c_ * cell + cell // 2, r_ * cell + cell // 2
        sw = rnd.uniform(0.93, 1.07); sr = rnd.uniform(0.9, 1.1); flip = rnd.random() < 0.5
        W = it['w'] * scale * sw                                        # одинаковый масштаб для всех: относительные размеры сохраняются
        pts = [((g - 0.5) * W, y / it['w'] * W * sr) for g, y in zip(it['g'], it['y'])]
        if flip: pts = [(-x, y) for x, y in pts][::-1]
        top = min(p[1] for p in pts); hgt = 1.05 * W                        # высота от вершины до нижнего края не меньше ширины
        P = [(cx + x, cy - hgt / 2 + (y - top)) for x, y in pts]            # вершина на cy - hgt/2, низ на cy + hgt/2
        bottom = cy + hgt / 2
        poly = P + [(P[-1][0], bottom), (P[0][0], bottom)]
        d.polygon(poly, fill=(222, 222, 222)); d.line(P, fill=(95, 95, 95), width=11, joint='curve')
    im.save(os.path.join(a.out, f'дуги_лист_{sh + 1}.png')); print('лист', sh + 1)
json.dump([dict(w=i['w'], rise=i['rise'], apex=i['apex'], tilt=i['tilt']) for i in sel], open(os.path.join(a.out, 'выбранные_дуги.json'), 'w'))
