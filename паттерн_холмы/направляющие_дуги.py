# Из дуг, снятых с образца (дуги.json), делает направляющие листы для Gemini: 2×3 силуэта холмиков (клетки-портреты)
# (серая линия = дуга, светло-серая область = ЦЕЛЫЙ холмик с длинной юбкой, 1.5 ширины). Кривые сглаживаются и слегка варьируются
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
S = a.size; cw, ch = S // 3, S // 2; per = 6; sheets = (len(sel) + per - 1) // per


def extend_right(P, W):
    """Продолжает кривую вправо вниз с тем же (и не меньшим) изгибом, пока касательная не станет почти вертикальной."""
    P = np.array(P, float); n = len(P)
    ang = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])); seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    m = max(3, n // 4); kap = (ang[-1] - ang[-1 - m]) / max(1e-6, seg[-m:].sum())      # кривизна на последней четверти
    kap = max(kap, 1.6 / W)                                                          # не меньше: дуга обязана загибаться вниз
    th = ang[-1]; x, y = P[-1]; out = []; ds = 2.0; L = 0
    while th < np.radians(86) and L < 1.1 * W:
        th += kap * ds; x += np.cos(th) * ds; y += np.sin(th) * ds; L += ds; out.append((x, y))
    return np.vstack([P, out]) if out else P


def mound(it, W, sr):
    """Полный купол: сглаженная дуга (как снята с образца) + продолжение вниз по обе стороны до вертикали. Возвращает точки контура сверху (слева направо)."""
    pts = np.array([((g - 0.5) * W, y / it['w'] * W * sr) for g, y in zip(it['g'], it['y'])])
    R = extend_right(pts, W)
    Lm = extend_right(pts[::-1] * np.array([-1, 1]), W)                               # левый конец: зеркалим, продолжаем, зеркалим назад
    Lm = (Lm * np.array([-1, 1]))[::-1]
    return np.vstack([Lm[:len(Lm) - len(pts)], R])                                    # левое продолжение + дуга + правое продолжение


shapes = []
for it in sel:
    sr = rnd.uniform(0.9, 1.1); sw = rnd.uniform(0.93, 1.07); flip = rnd.random() < 0.5
    W = it['w'] * sw; P = mound(it, W, sr)
    if flip: P = (P * np.array([-1, 1]))[::-1]
    top = P[:, 1].min(); bw = P[:, 0].max() - P[:, 0].min()
    base = max(top + 1.45 * W, P[:, 1].max() + 0.25 * W)                              # основание: ниже плеч, юбка с прямыми боками
    shapes.append(dict(P=P, base=base, bw=bw, hh=base - top, top=top))
scale = min(0.88 * cw / max(sh['bw'] for sh in shapes), 0.92 * ch / max(sh['hh'] for sh in shapes))
for sh_ in range(sheets):
    im = Image.new('RGB', (S, S), (255, 255, 255)); d = ImageDraw.Draw(im)
    for k, sp in enumerate(shapes[sh_ * per:sh_ * per + per]):
        r_, c_ = divmod(k, 3); cx, cy = c_ * cw + cw // 2, r_ * ch + ch // 2
        P = sp['P'] * scale; top = sp['top'] * scale; hh = sp['hh'] * scale
        xm = (P[:, 0].min() + P[:, 0].max()) / 2
        Q = [(cx + x - xm, cy - hh / 2 + (y - top)) for x, y in P]; bottom = cy + hh / 2
        Q = [(Q[0][0], bottom)] + Q + [(Q[-1][0], bottom)]                  # линия идёт по верху, плечам и вниз по обеим сторонам до основания: оба конца на одной горизонтали
        d.polygon(Q, fill=(222, 222, 222)); d.line(Q, fill=(95, 95, 95), width=11, joint='curve')
    im.save(os.path.join(a.out, f'дуги_лист_{sh_ + 1}.png')); print('лист', sh_ + 1)
json.dump([dict(w=i['w'], rise=i['rise'], apex=i['apex'], tilt=i['tilt']) for i in sel], open(os.path.join(a.out, 'выбранные_дуги.json'), 'w'))
