# Сборка «сетки холмов»: нижний ряд закрывает верхний; полоса каждого холмика обрезается в той точке, где её накрывает
# следующий холмик, поэтому конец полосы входит под соседа и вливается в его тёмную дугу, торчащих хвостов нет.
# Бесшовно: x заворачивается, по y три копии плитки подряд (видна средняя).
#   python -I сборка_сетка.py --elems спрайты --w 2175 --h 2175 --cols 3 --overlap 1.75 --rowfrac 0.15 --out плитка.png
import argparse, glob, math, os, random
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--elems', default='спрайты'); ap.add_argument('--out', default='плитка_сетка.png')
ap.add_argument('--w', type=int, default=2175); ap.add_argument('--h', type=int, default=2175); ap.add_argument('--cols', type=int, default=3)
ap.add_argument('--overlap', type=float, default=1.75); ap.add_argument('--rowfrac', type=float, default=0.15)
ap.add_argument('--jx', type=float, default=0.05); ap.add_argument('--jy', type=float, default=0.05); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--tone', type=float, default=0.0); ap.add_argument('--bgcol', default='1f4358'); ap.add_argument('--dive', type=int, default=10)
a = ap.parse_args(); W, H = a.w, a.h; rnd = random.Random(a.seed)
lib = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(a.elems, '*.npz')))]; assert lib
px = W / a.cols; rows = max(2, round(H / (px * a.overlap * a.rowfrac))); py = H / rows
print(f'спрайтов {len(lib)}; шаг x {px:.0f}, рядов {rows}, шаг y {py:.0f}')
used = [0] * len(lib); pl = []
def tdist(x1, y1, x2, y2):
    dx = abs(x1 - x2); dx = min(dx, W - dx); dy = abs(y1 - y2); dy = min(dy, H - dy); return math.hypot(dx, dy)
for r in range(rows):
    sh = (r % 2) * px / 2
    for c in range(a.cols):
        x = (c * px + sh + rnd.uniform(-a.jx, a.jx) * px) % W; y = r * py + rnd.uniform(-a.jy, a.jy) * py
        cand = sorted(range(len(lib)), key=lambda i: (used[i], rnd.random())); pick = None
        for i in cand:
            if all(not (q[3] == i and tdist(x, y % H, q[1], q[2] % H) < 1.9 * px) for q in pl): pick = i; break
        pick = cand[0] if pick is None else pick; used[pick] += 1
        pl.append((r, x, y, pick, rnd.random() < 0.5, 1 + rnd.uniform(-a.tone, a.tone)))

def inst(p):
    r, x, y, i, fl, tone = p; d = lib[i]; fill, stroke, nmap, C = d['fill'].copy(), d['stroke'].copy(), d['nmap'].copy(), d['C'].copy()
    w = fill.shape[1]
    if fl:
        fill, stroke, nmap = fill[:, ::-1], stroke[:, ::-1], nmap[:, ::-1]; C[:, 0] = w - 1 - C[:, 0]; C = C[::-1]; nmap = (len(C) - 1 - nmap).astype(np.int16)
    if tone != 1:
        for L in (fill, stroke): L[..., :3] = np.clip(L[..., :3].astype(float) * tone, 0, 255).astype(np.uint8)
    return dict(fill=np.ascontiguousarray(fill), stroke=np.ascontiguousarray(stroke), nmap=np.ascontiguousarray(nmap), C=C, apex=int(np.argmin(C[:, 1])))

I = [inst(p) for p in pl]; N = len(pl)
# экземпляры: (порядок, номер, смещение по y в высокой канве). канва: 4H высоты, копия kc=-1 -> y0 = y, kc=0 -> y+H, kc=+1 -> y+2H
TH = 4 * H; order = []
for kc in (0, 1, 2):
    for n in range(N): order.append((kc, n))
def pos(n, kc):
    r, x, y, i, fl, tone = pl[n]; h, w = I[n]['fill'].shape[:2]
    return int(round(x - w / 2)), int(round(y + kc * H - 0)), w, h

owner = np.full((TH, W), -1, np.int16)
for k, (kc, n) in enumerate(order):
    x0, y0, w, h = pos(n, kc); al = np.maximum(I[n]['fill'][..., 3], I[n]['stroke'][..., 3]) > 128
    for kx in (-1, 0, 1):
        xx = x0 + kx * W
        if xx >= W or xx + w <= 0 or y0 >= TH or y0 + h <= 0: continue
        sx0, sy0 = max(0, -xx), max(0, -y0); sx1, sy1 = min(w, W - xx), min(h, TH - y0)
        sub = owner[y0 + sy0:y0 + sy1, xx + sx0:xx + sx1]; sub[al[sy0:sy1, sx0:sx1]] = k

canvas = np.zeros((TH, W, 4), np.uint8)
def blit(layer, x0, y0):
    h, w = layer.shape[:2]
    for kx in (-1, 0, 1):
        xx = x0 + kx * W
        if xx >= W or xx + w <= 0 or y0 >= TH or y0 + h <= 0: continue
        sx0, sy0 = max(0, -xx), max(0, -y0); sx1, sy1 = min(w, W - xx), min(h, TH - y0)
        yield xx, sx0, sy0, sx1, sy1
bgc = tuple(int(a.bgcol[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
cv = Image.new('RGBA', (W, TH), bgc)
cuts = []
for k, (kc, n) in enumerate(order):
    d = I[n]; x0, y0, w, h = pos(n, kc); C = d['C']; ap_ = d['apex']; lo, hi = 0, len(C) - 1
    def covered(t):
        X = int(round(C[t, 0])) + x0; Y = int(round(C[t, 1])) + y0
        return 0 <= Y < TH and owner[Y, X % W] > k
    cl, cr = 0, len(C) - 1
    for t in range(ap_, len(C)):
        if covered(t): cr = min(len(C) - 1, t + a.dive); break
    for t in range(ap_, -1, -1):
        if covered(t): cl = max(0, t - a.dive); break
    cuts.append((cl, cr))
    stroke = d['stroke'].copy(); nm = d['nmap']; stroke[..., 3] = np.where((nm >= cl) & (nm <= cr), stroke[..., 3], 0)
    for layer in (d['fill'], stroke):
        L = Image.fromarray(layer)
        for xx, sx0, sy0, sx1, sy1 in blit(layer, x0, y0):
            if y0 + sy1 <= 0: continue
            cv.alpha_composite(L, (xx + sx0, y0 + sy0), (sx0, sy0, sx1, sy1))
out = np.array(cv.convert('RGB'))[H:2 * H]            # средняя копия: видна верхняя и нижняя соседние, швов нет
Image.fromarray(out).save(a.out)
inner = (np.abs(out[:, 1:].astype(int) - out[:, :-1]).mean() + np.abs(out[1:].astype(int) - out[:-1]).mean()) / 2
big = np.tile(out, (2, 2, 1)).astype(int)
sx = np.abs(big[:, W] - big[:, W - 1]).mean(); sy = np.abs(big[H] - big[H - 1]).mean()
hole = (np.abs(out.astype(int) - np.array(bgc[:3])).max(axis=2) < 4).mean()
print('внутри %.2f; шов по x %.2f; шов по y %.2f; пикселей цвета фона (дыры) %.4f%%' % (inner, sx, sy, 100 * hole))
stem = os.path.splitext(a.out)[0]
Image.fromarray(big.astype(np.uint8)).resize((1600, 1600), Image.LANCZOS).save(stem + '_2x2.jpg', quality=90)
c = 700; Image.fromarray(big.astype(np.uint8)).crop((W - c, H - c, W + c, H + c)).save(stem + '_стык_центр.jpg', quality=92)
