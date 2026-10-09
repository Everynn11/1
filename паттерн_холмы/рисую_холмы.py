# Процедурные холмики для сетки (вместо листов Gemini): форма = снятая с образца дуга (дуги.json) + плечи вниз (как в направляющих),
# тёмная полоса ТОЛЬКО по верху и плечам (с сужением к концам и неровным краем), под ней 1-3 светлые линии, заливка с мягкой акварельной текстурой.
# Спрайты кладутся в папку, дальше обычная сборка: python -I сборка_холмов.py --elems спрайты ...
#   python -I рисую_холмы.py --arcs дуги.json --out спрайты --n 24 --width 634
import argparse, json, math, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
ap = argparse.ArgumentParser(); ap.add_argument('--arcs', default='дуги.json'); ap.add_argument('--out', default='спрайты'); ap.add_argument('--n', type=int, default=24)
ap.add_argument('--width', type=int, default=634); ap.add_argument('--bandw', type=float, default=60.0); ap.add_argument('--maxang', type=float, default=80.0); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--fill', default='567783'); ap.add_argument('--band', default='183442'); ap.add_argument('--ss', type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); SS = a.ss
hexc = lambda h: np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
FILL, BAND = hexc(a.fill), hexc(a.band); WHITE, MINT = np.array([244, 246, 244.]), np.array([190, 216, 208.])

# ---- дуги: сглаженный горб с одной вершиной
arcs = []
for c in json.load(open(a.arcs)):
    x = np.array(c['x']); y = np.array(c['y']); w = x.max() - x.min(); u = (x - x.min()) / w
    p = np.polyfit(u, y, 3); g = np.linspace(0, 1, 60); yy = np.polyval(p, g); yy -= yy.max()
    res = float(np.sqrt(np.mean((np.polyval(p, u) - y) ** 2)) / w); dy = np.diff(yy); sg = np.sign(dy[np.abs(dy) > 1e-3 * w])
    apex = float(g[yy.argmin()])
    if res > 0.05 or not (0.12 < apex < 0.88) or int(np.sum(sg[1:] != sg[:-1])) > 1 or -yy.min() < 0.07 * w or abs(yy[0] - yy.min()) < 0.03 * w or abs(yy[-1] - yy.min()) < 0.03 * w: continue
    arcs.append(dict(w=w, g=g, y=yy))
print('дуг годных:', len(arcs))


def extend_right(P, W):
    P = np.array(P, float); n = len(P)
    ang = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])); seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    m = max(3, n // 4); kap = max((ang[-1] - ang[-1 - m]) / max(1e-6, seg[-m:].sum()), 1.6 / W)
    th = ang[-1]; x, y = P[-1]; out = []; L = 0
    while th < np.radians(86) and L < 1.1 * W:
        th += kap * 2; x += math.cos(th) * 2; y += math.sin(th) * 2; L += 2; out.append((x, y))
    return np.vstack([P, out]) if out else P


def contour(arc, W):
    pts = np.array([((g - 0.5) * W, yv / arc['w'] * W) for g, yv in zip(arc['g'], arc['y'])])
    R = extend_right(pts, W); Lm = extend_right(pts[::-1] * np.array([-1, 1]), W); Lm = (Lm * np.array([-1, 1]))[::-1]
    return np.vstack([Lm[:len(Lm) - len(pts)], R])


def convex_top(P):
    """Верхняя выпуклая оболочка контура: убирает вогнутости и перегибы по верху, оставляет округлый купол."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(P); v = list(h.vertices); pts = P[v]
    l = int(np.argmin(pts[:, 0])); r = int(np.argmax(pts[:, 0])); t = int(np.argmin(pts[:, 1])); n = len(v)
    def walk(a_, b_, step):
        out = [a_]
        while out[-1] != b_: out.append((out[-1] + step) % n)
        return out
    ch = walk(l, r, 1)
    if t not in ch: ch = walk(l, r, -1)
    return pts[ch]


def resample(P, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], d[-1]


def fbm(h, w, scales=(70, 26, 9), amps=(1, .55, .3)):
    o = np.zeros((h, w))
    for s_, am in zip(scales, amps):
        gh, gw = max(3, h // s_ + 3), max(3, w // s_ + 3); gz = ndi.zoom(rng.normal(0, 1, (gh, gw)), (h / gh, w / gw), order=3)
        o += am * gz[:h, :w]
    return o / sum(amps)


def stroke(shape, C, widths, soft=0.0):
    """Полигон вдоль кривой C с переменной шириной; возвращает маску 0..1."""
    t = np.gradient(C, axis=0); t /= np.maximum(1e-6, np.hypot(*t.T))[:, None]; nrm = np.c_[-t[:, 1], t[:, 0]]
    poly = [tuple(p) for p in C + nrm * (widths / 2)[:, None]] + [tuple(p) for p in (C - nrm * (widths / 2)[:, None])[::-1]]
    im = Image.new('L', (shape[1], shape[0]), 0); ImageDraw.Draw(im).polygon(poly, fill=255); return np.array(im).astype(float) / 255, nrm


def make_sprite(arc, idx):
    W = a.width * SS; P = contour(arc, W)
    P, _ = resample(convex_top(P), 2.0 * SS); P = ndi.gaussian_filter1d(P, sigma=12, axis=0, mode='nearest')      # выпуклый и скруглённый верх
    top = P[:, 1].min(); bw = P[:, 0].max() - P[:, 0].min()
    base = max(top + 1.5 * W * 0.55, P[:, 1].max() + 0.25 * W); pad = int(0.08 * W)
    x0 = P[:, 0].min() - pad; Hs = int(base - top + 2 * pad); Ws = int(bw + 2 * pad)
    Q = P - np.array([x0, top - pad])
    fm = Image.new('L', (Ws, Hs), 0); ImageDraw.Draw(fm).polygon([tuple(p) for p in Q] + [(Q[-1][0], Hs - pad), (Q[0][0], Hs - pad)], fill=255); fmask = np.array(fm).astype(float) / 255
    tone = 1 + rng.uniform(-0.03, 0.03)
    # заливка: мягкие пятна + зерно + светлее к низу
    f = fbm(Hs, Ws); vert = np.linspace(0, 1, Hs)[:, None]
    col = FILL[None, None, :] * tone * (1 + 0.07 * f[..., None] + 0.05 * vert[..., None]) + rng.normal(0, 1.6, (Hs, Ws, 1))
    # полоса и линии по кривой
    C, L = resample(Q, 3.0 * SS / 2); s = np.linspace(0, 1, len(C))
    C = ndi.gaussian_filter1d(C, sigma=7, axis=0, mode='nearest')
    _t = np.gradient(C, axis=0); _ang = np.degrees(np.arctan2(np.abs(_t[:, 1]), np.maximum(1e-6, np.abs(_t[:, 0]))))   # крутизна касательной
    _top = int(np.argmin(C[:, 1])); _l = _top; _r = _top
    while _l > 0 and _ang[_l] < a.maxang: _l -= 1
    while _r < len(C) - 1 and _ang[_r] < a.maxang: _r += 1
    C = C[_l:_r + 1]; s = np.linspace(0, 1, len(C))                        # полоса только по верху и пологой части плеч, без хвостов вниз            # гладкая кривая: без изломов от разреженных точек дуги
    taper = np.clip((np.minimum(s, 1 - s) / 0.22) ** 0.9, 0.62, 1)
    tilt = 1 + 0.28 * (2 * s - 1) * rng.choice([-1, 1])
    bw_s = a.bandw * SS * taper * tilt                                          # гладкая ширина (по ней идут светлые линии)
    bw_ = bw_s * (1 + 0.07 * ndi.gaussian_filter(rng.normal(0, 1, len(C)), 25) / 0.1)   # у самой полосы лёгкая живая неровность, без зубцов
    # светлые линии: 1-3, сразу под тёмной полосой, поверх заливки
    nlines = int(rng.choice([1, 2, 2, 3])); lw = 20 * SS; lrgb = np.zeros((Hs, Ws, 3)); la = np.zeros((Hs, Ws))
    for k in range(nlines):
        off = bw_s * 0.5 + lw * (0.55 + 1.05 * k)
        wob = ndi.gaussian_filter(rng.normal(0, 1, len(C)), 8) * 0.18 * lw
        t_ = np.gradient(C, axis=0); t_ /= np.maximum(1e-6, np.hypot(*t_.T))[:, None]; nm = np.c_[-t_[:, 1], t_[:, 0]]
        Ck = C + nm * (off + wob)[:, None]
        sl = (s > 0.14 + 0.04 * k) & (s < 0.86 - 0.04 * k); Ck = Ck[sl]
        wk = lw * np.clip((np.minimum(np.linspace(0, 1, len(Ck)), 1 - np.linspace(0, 1, len(Ck))) / 0.12) ** 0.7, 0.2, 1)
        m, _ = stroke((Hs, Ws), Ck, wk); m = ndi.gaussian_filter(m, 0.7 * SS / 2) * fmask
        colr = WHITE if k % 2 == 0 else MINT; lrgb = lrgb * (1 - m[..., None]) + colr * m[..., None]; la = la + m * (1 - la)
    # тёмная полоса с неровным краем
    bm, _ = stroke((Hs, Ws), C, bw_)
    b = ndi.gaussian_filter(bm, 1.3 * SS / 2); nz = ndi.gaussian_filter(rng.normal(0, 1, (Hs, Ws)), 5.0 * SS / 2) * 4.0
    bmask = np.clip((b - 0.5 + 0.045 * nz) / 0.18 + 0.5, 0, 1)
    bcol = BAND[None, None, :] * (1 + 0.05 * f[..., None]) + rng.normal(0, 1.2, (Hs, Ws, 1))
    fill_a = ndi.gaussian_filter(fmask, 0.8 * SS / 2)
    sa = 1 - (1 - la) * (1 - bmask); srgb = (lrgb * la[..., None] * (1 - bmask[..., None]) + bcol * bmask[..., None]) / np.maximum(sa, 1e-4)[..., None]
    def down(rgb, al):
        im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), al * 255]).astype(np.uint8), 'RGBA'); return im.resize((max(2, round(Ws / SS)), max(2, round(Hs / SS))), Image.LANCZOS)
    fl, st = down(col, fill_a), down(srgb, sa)
    Cn = C / SS; h2, w2 = fl.size[1], fl.size[0]
    # карта «индекс ближайшей точки кривой» по пикселям (нужна сборке, чтобы обрезать полосу там, где её накрывает сосед)
    dens = np.c_[np.interp(np.arange(0, len(Cn) - 1, 0.25), np.arange(len(Cn)), Cn[:, 0]), np.interp(np.arange(0, len(Cn) - 1, 0.25), np.arange(len(Cn)), Cn[:, 1])]
    lab = np.zeros((h2, w2), np.int32); xi = np.clip(np.round(dens[:, 0]).astype(int), 0, w2 - 1); yi = np.clip(np.round(dens[:, 1]).astype(int), 0, h2 - 1)
    lab[yi, xi] = (np.arange(len(dens)) * 0.25 + 1).astype(np.int32)
    idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    nmap = (lab[idx[0], idx[1]] - 1).astype(np.int16)
    return dict(fill=np.array(fl), stroke=np.array(st), nmap=nmap, C=Cn.astype(np.float32))


for i in range(a.n):
    d = make_sprite(arcs[i % len(arcs)], i)
    np.savez_compressed(os.path.join(a.out, f'P{i + 1:02d}.npz'), **d)
    Image.alpha_composite(Image.fromarray(d['fill']), Image.fromarray(d['stroke'])).save(os.path.join(a.out, f'P{i + 1:02d}.png'))
print('спрайтов', a.n, 'размер первого', d['fill'].shape[:2])
