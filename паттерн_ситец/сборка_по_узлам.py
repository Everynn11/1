# Плитка из лент на «безумной» геометрии: лозы под разными замыкающимися углами, на каждой своя лента.
# Лента на каждой линии масштабируется так, чтобы целое число периодов ровно укладывалось в замкнутую петлю
# (длина петли = R·√(p²+q²)), поэтому лозы замыкаются без шва. В пересечениях лозы идут поочерёдно над/под.
#   python сборка_геометрия.py --out плитка.png --size 2048
import argparse, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='плитка_геометрия.png')
ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--period', type=float, default=99999)      # 99999 = на каждую лозу одна лента = вся петля (n=1)
ap.add_argument('--margin', type=int, default=36)          # нахлёст соседних периодов, px (убирает светлые швы)
ap.add_argument('--dir', default='лента')
ap.add_argument('--bg', default='')
ap.add_argument('--cut', type=int, default=46)             # радиус перерисовки «над» вокруг узла, px
a = ap.parse_args()

R = a.size
# линия: (p, q) — сколько сторон плитки проходит лоза по x и y за петлю; c — смещение (доля R); для вертикали x0
import json
PH = json.load(open('узлы/сдвиги_подобраны.json', encoding='utf8'))     # ph и зеркало каждой ленты, подбор: узлы/подбор_сдвигов.py
LINES = [
    dict(p=1, q=1, c=0.00, rib='по_узлам_A', flip=PH['A']['flip'], ph=PH['A']['ph']),
    dict(p=1, q=-1, c=0.40, rib='по_узлам_B', flip=PH['B']['flip'], ph=PH['B']['ph']),
    dict(p=1, q=2, c=0.15, rib='по_узлам_C', flip=PH['C']['flip'], ph=PH['C']['ph']),
    dict(p=2, q=1, c=0.62, rib='по_узлам_D', flip=PH['D']['flip'], ph=PH['D']['ph']),
    dict(p=0, q=1, c=0.00, x0=0.55, rib='по_узлам_E', flip=PH['E']['flip'], ph=PH['E']['ph']),
]


def vine_only(im):
    arr = np.array(im)
    x = 8
    ys = np.where(arr[:, x, 3] > 200)[0]
    groups = np.split(ys, np.where(np.diff(ys) > 3)[0] + 1)
    g = groups[-1]
    sy = int(g.mean())
    seed = arr[sy - 2:sy + 3, x - 2:x + 3, :3].reshape(-1, 3).mean(axis=0)
    d = np.abs(arr[..., :3].astype(int) - seed[None, None, :]).max(axis=2)
    cand = (d < 28) & (arr[..., 3] > 200)
    lab, _ = ndi.label(cand, structure=np.ones((3, 3)))
    m = lab == lab[sy, x]
    m = ndi.binary_dilation(m, iterations=3) & (arr[..., 3] > 40)
    out = arr.copy(); out[..., 3] = np.where(m, arr[..., 3], 0)
    return Image.fromarray(out)


def load(name):
    im = Image.open(os.path.join(a.dir, name + '.png')).convert('RGBA')
    return im, vine_only(im)


def piece(im, T, flip, m):
    """Один период ленты (ширина T+2m): берётся из утроенной ленты, поэтому края — внутренние пиксели."""
    W0, H0 = im.size
    tri = Image.new('RGBA', (W0 * 3, H0), (0, 0, 0, 0))
    for i in range(3):
        tri.alpha_composite(im, (i * W0, 0))
    k = T / W0
    tri = tri.resize((int(round(W0 * 3 * k)), int(round(H0 * k))), Image.LANCZOS)
    t = int(round(T))
    p = tri.crop((t - m, 0, 2 * t + m, tri.height))
    return p.transpose(Image.FLIP_TOP_BOTTOM) if flip else p


def paste_wrap(dst, im, cx, cy):
    w, h = im.size
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    for kx in range(-2, 3):
        for ky in range(-2, 3):
            xx, yy = x0 + kx * R, y0 + ky * R
            if xx >= R or yy >= R or xx + w <= 0 or yy + h <= 0:
                continue
            sx0, sy0 = max(0, -xx), max(0, -yy)
            sx1, sy1 = min(w, R - xx), min(h, R - yy)
            dst.alpha_composite(im, (xx + sx0, yy + sy0), (sx0, sy0, sx1, sy1))


cache = {}
layers = []
for i, L in enumerate(LINES):
    if L['rib'] not in cache:
        cache[L['rib']] = load(L['rib'])
    im, vo = cache[L['rib']]
    loop = R * math.hypot(L['p'], L['q'])
    n = max(1, round(loop / a.period))
    T = loop / n
    deg = math.degrees(math.atan2(L['q'], L['p']))
    d = np.array([L['p'], L['q']], float) / math.hypot(L['p'], L['q'])
    start = np.array([L.get('x0', 0.0) * R, L['c'] * R])
    full = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    vin = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    pc, pv = piece(im, T, L['flip'], a.margin), piece(vo, T, L['flip'], a.margin)
    rc, rv = pc.rotate(-deg, resample=Image.BICUBIC, expand=True), pv.rotate(-deg, resample=Image.BICUBIC, expand=True)
    for j in range(n):
        u = (j + 0.5 + L['ph']) * T
        c = (start + d * u) % R
        paste_wrap(full, rc, c[0], c[1])
        paste_wrap(vin, rv, c[0], c[1])
    layers.append((full, vin))
    print(f"лоза {i + 1}: ({L['p']},{L['q']}) {L['rib']}: петля {loop:.0f}px = {n} периода по {T:.0f}px, угол {deg:.1f}°")

canvas = Image.new('RGBA', (R, R), (0, 0, 0, 0))
for full, vin in layers:
    canvas.alpha_composite(full)

# узлы: области, где пересекаются стебли двух разных лоз; победителя чередуем
va = [np.array(v)[..., 3] > 100 for _, v in layers]
node_list = []
for i in range(len(layers)):
    for j in range(i + 1, len(layers)):
        ov = ndi.binary_dilation(va[i] & va[j], iterations=2)
        lab, nn = ndi.label(ov)
        for k in range(1, nn + 1):
            mk = lab == k
            if mk.sum() < 30:
                continue
            ys, xs = np.nonzero(mk)
            node_list.append((i, j, xs.mean(), ys.mean()))
print('узлов пересечения:', len(node_list))
yy, xx = np.mgrid[0:R, 0:R]
for idx, (i, j, x, y) in enumerate(sorted(node_list, key=lambda t: (round(t[3] / 150), t[2]))):
    win = i if idx % 2 == 0 else j
    if win == len(layers) - 1 or win > max(i, j) - 1 and win == max(i, j):
        continue                                           # победитель уже сверху по порядку наложения
    dx = np.minimum(np.abs(xx - x), R - np.abs(xx - x)); dy = np.minimum(np.abs(yy - y), R - np.abs(yy - y))
    soft = np.clip((a.cut - np.hypot(dx, dy)) / 12.0, 0, 1)
    patch = np.array(layers[win][1])
    patch[..., 3] = (patch[..., 3] * soft).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(patch))

bgc = tuple(int(a.bg[k:k + 2], 16) for k in (0, 2, 4)) + (255,) if a.bg else (255, 255, 255, 255)
out = Image.new('RGBA', (R, R), bgc)
out.alpha_composite(canvas)
out.convert('RGB').save(a.out)
canvas.save(os.path.splitext(a.out)[0] + '_прозр.png')
t = Image.new('RGB', (R * 2, R * 2))
for x_ in (0, 1):
    for y_ in (0, 1):
        t.paste(out.convert('RGB'), (x_ * R, y_ * R))
t.resize((1600, 1600), Image.LANCZOS).save(os.path.splitext(a.out)[0] + '_2x2.jpg', quality=90)
o = np.array(out.convert('RGB')).astype(float)
inner = (np.abs(o[:, 1:] - o[:, :-1]).mean() + np.abs(o[1:] - o[:-1]).mean()) / 2
print('перепад внутри %.2f; стык лево-право %.2f, верх-низ %.2f' % (inner, np.abs(o[:, -1] - o[:, 0]).mean(), np.abs(o[-1] - o[0]).mean()))
print('готово', a.out)
