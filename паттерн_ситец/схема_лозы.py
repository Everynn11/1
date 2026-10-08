# Идеальная бесшовная схема для раскраски в Gemini: переплетённые лозы (линии) + цветные метки цветов и листьев.
# Всё считается на торе, поэтому края стыкуются точно.
#   python схема_лозы.py --out схема_лозы.png --size 2048
import argparse, math
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='схема_лозы.png')
ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--amp', type=float, default=0.05)      # амплитуда волны лозы, доля размера
ap.add_argument('--waves', type=int, default=2)         # волн на период
ap.add_argument('--flowers', type=int, default=6)       # цветов на одну лозу
ap.add_argument('--stems-only', action='store_true')    # только лозы: без цветов, листьев и побегов
ap.add_argument('--gap', type=float, default=0.012)     # зазор между метками, доля размера
a = ap.parse_args()

R = a.size
SS = 2
rng = np.random.default_rng(a.seed)
W = R * SS


def wrap(v):
    return v % R


# ------------------------------------------------------------ лозы
def vine(direction, c, phi):
    """Центральная линия лозы: точки (x, y) на торе. direction = +1 «/», -1 «\\»."""
    s = np.linspace(0, R, 3000, endpoint=False)
    y = direction * s + c + a.amp * R * np.sin(2 * np.pi * a.waves * s / R + phi)
    return np.stack([s % R, y % R], 1), s


VINES = []
for d, cs in ((+1, (0.0, 0.5)), (-1, (0.25, 0.75))):
    for ci, c in enumerate(cs):
        VINES.append(dict(d=d, c=c * R, phi=rng.uniform(0, 2 * np.pi) if False else (0.6 + 1.9 * ci + (0.7 if d < 0 else 0))))
for v in VINES:
    v['pts'], v['s'] = vine(v['d'], v['c'], v['phi'])
print('лоз:', len(VINES))


def tangent(v, i):
    p0, p1 = v['pts'][(i - 1) % len(v['pts'])], v['pts'][(i + 1) % len(v['pts'])]
    d = p1 - p0
    d = (d + R / 2) % R - R / 2
    n = np.hypot(*d)
    return d / n


def point_at(v, s_target):
    i = int((s_target % R) / R * len(v['pts'])) % len(v['pts'])
    return v['pts'][i], tangent(v, i)


# ------------------------------------------------------------ рисование на торе
img = Image.new('RGB', (W, W), (255, 255, 255))
dr = ImageDraw.Draw(img)


def tor_shifts():
    for dx in (-W, 0, W):
        for dy in (-W, 0, W):
            yield dx, dy


def poly(points, fill, width=0, outline=None, close=True):
    pts = [(x * SS, y * SS) for x, y in points]
    for dx, dy in tor_shifts():
        sp = [(x + dx, y + dy) for x, y in pts]
        if close:
            dr.polygon(sp, fill=fill, outline=outline)
        else:
            dr.line(sp, fill=fill, width=width, joint='curve')


def stroke(points, width, fill):
    """Гладкая линия: круги диаметром width вдоль ломаной (частый шаг), на торе."""
    pts = np.array(points, float)
    dense = [pts[0]]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(p1 - p0)) / 0.6))
        for k in range(1, n + 1):
            dense.append(p0 + (p1 - p0) * k / n)
    r = width / 2
    for dx, dy in tor_shifts():
        for x, y in dense:
            X, Y = x * SS + dx, y * SS + dy
            if -r * SS <= X <= W + r * SS and -r * SS <= Y <= W + r * SS:
                dr.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=fill)


def circle(cx, cy, r, fill):
    for dx, dy in tor_shifts():
        x, y = cx * SS + dx, cy * SS + dy
        dr.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=fill)


def ellipse_poly(cx, cy, rl, rw, ang, n=24, pointy=False):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rl * np.cos(t), rw * np.sin(t)
    if pointy:                                   # заострённый лист: сжать к концам
        x = rl * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.8
        y = rw * np.sin(t) * (1 - 0.35 * np.abs(np.cos(t)))
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + x[i] * ca - y[i] * sa, cy + x[i] * sa + y[i] * ca) for i in range(n)]


def star(cx, cy, r, ang, spikes=11):
    pts = []
    for i in range(spikes * 2):
        rr = r if i % 2 == 0 else r * 0.78
        t = ang + math.pi * i / spikes
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    return pts


VINE_COL = (112, 96, 58)
THICK = 0.011 * R

# ------------------------------------------------------------ цветы и листья (метки)
# тип: (цвет метки, радиус доля R, форма)
TYPES = [
    ('роза тёмная',   (176, 48, 96),   0.065, 'circle'),
    ('роза розовая',  (240, 140, 160), 0.062, 'circle'),
    ('пион кремовый', (246, 228, 208), 0.072, 'circle'),
    ('василёк',       (58, 95, 205),   0.048, 'star'),
    ('гвоздика',      (255, 111, 160), 0.044, 'star'),
    ('бутон',         (232, 160, 168), 0.032, 'bud'),
]
LEAF_DARK, LEAF_LIGHT = (70, 120, 70), (143, 175, 136)

placed = []                                      # (x, y, r) занятые под цветами


def free(x, y, r):
    for px, py, pr in placed:
        dx, dy = abs(x - px), abs(y - py)
        dx, dy = min(dx, R - dx), min(dy, R - dy)
        if math.hypot(dx, dy) < (r + pr) * R * 1.0 + a.gap * R:
            return False
    return True


flowers, leaves = [], []
tp = 0
for vi, v in enumerate(VINES):
    n = a.flowers
    for k in range(n):
        for attempt in range(40):
            s = (k + 0.5 + rng.uniform(-0.25, 0.25)) * R / n + vi * 41
            p, t = point_at(v, s)
            nrm = np.array([-t[1], t[0]])
            side = 1 if (k + vi) % 2 == 0 else -1
            if attempt % 2:
                side = -side
            name, col, rr, shape = TYPES[(tp + attempt // 2) % len(TYPES)]
            off = (0.045 + rr + rng.uniform(0.0, 0.03)) * R
            cx, cy = p + side * nrm * off
            if free(cx % R, cy % R, rr):
                placed.append((cx % R, cy % R, rr))
                flowers.append(dict(x=cx, y=cy, r=rr * R, col=col, shape=shape, vx=p[0], vy=p[1], name=name, vi=vi))
                tp += 1
                break

# листья вдоль лоз
for vi, v in enumerate(VINES):
    step = 0.038 * R
    s = rng.uniform(0, step)
    side = 1
    while s < R:
        p, t = point_at(v, s)
        nrm = np.array([-t[1], t[0]])
        ll = rng.uniform(0.075, 0.105) * R
        ang = math.atan2(t[1], t[0]) + side * rng.uniform(0.7, 1.05)
        cx = p[0] + math.cos(ang) * ll * 0.55
        cy = p[1] + math.sin(ang) * ll * 0.55
        ok = True
        for fx in flowers:
            dx, dy = abs(cx - fx['x']) % R, abs(cy - fx['y']) % R
            dx, dy = min(dx, R - dx), min(dy, R - dy)
            if math.hypot(dx, dy) < fx['r'] + ll * 0.5:
                ok = False
        if ok:
            leaves.append(dict(x=cx, y=cy, rl=ll / 2, rw=ll * 0.2, ang=ang, dark=rng.random() < 0.5, px=p[0], py=p[1]))
        s += step * rng.uniform(0.8, 1.4)
        side = -side

if a.stems_only:
    flowers, leaves = [], []

# листья (под лозой и цветами)
for lf in leaves:
    poly(ellipse_poly(lf['x'], lf['y'], lf['rl'], lf['rw'], lf['ang'], pointy=True), LEAF_DARK if lf['dark'] else LEAF_LIGHT)
    # средняя жилка-черенок к лозе
    poly([(lf['px'], lf['py']), (lf['x'], lf['y'])], VINE_COL, width=int(THICK * 0.45 * SS), close=False)

# стебли цветов
for fx in flowers:
    poly([(fx['vx'], fx['vy']), (fx['x'], fx['y'])], VINE_COL, width=int(THICK * 0.7 * SS), close=False)

# лозы: сначала все, потом узлы переплетения «над/под»
for v in VINES:
    pts = v['pts']
    seg = [(x, y) for x, y in pts]
    # разрыв на границе тора: рисуем по отрезкам
    path = [np.array(pts[0])]
    for i in range(1, len(pts) + 1):
        d = (pts[i % len(pts)] - path[-1] + R / 2) % R - R / 2
        path.append(path[-1] + d)
    stroke(path, THICK, VINE_COL)

# узлы пересечений: находим и поднимаем одну лозу над другой с белым ореолом
nodes = []
for i, vi_ in enumerate(VINES):
    for j, vj in enumerate(VINES):
        if j <= i or vi_['d'] == vj['d']:
            continue
        A, B = vi_['pts'], vj['pts']
        step = 6
        Ai, Bj = A[::step], B[::step]
        dx = np.abs(Ai[:, None, 0] - Bj[None, :, 0]); dx = np.minimum(dx, R - dx)
        dy = np.abs(Ai[:, None, 1] - Bj[None, :, 1]); dy = np.minimum(dy, R - dy)
        D = np.hypot(dx, dy)
        idx = np.argwhere(D < THICK * 0.9)
        pts_nodes = []
        for ia, ib in idx:
            q = (Ai[ia] + Bj[ib]) / 2
            if all(math.hypot(min(abs(q[0] - n_[0]), R - abs(q[0] - n_[0])), min(abs(q[1] - n_[1]), R - abs(q[1] - n_[1]))) > THICK * 3 for n_ in pts_nodes):
                pts_nodes.append((q[0], q[1], ia * step, ib * step))
        for n_ in pts_nodes:
            nodes.append((i, j, *n_))
print('узлов переплетения:', len(nodes))
for k, (i, j, x, y, ia, ib) in enumerate(nodes):
    top = i if k % 2 == 0 else j                      # чередуем, кто сверху
    v = VINES[top]
    idx0 = ia if top == i else ib
    step_px = R / len(v['pts'])

    def part(sp):
        span = int(sp * R / step_px)
        pts = [v['pts'][(idx0 + o) % len(v['pts'])] for o in range(-span, span + 1)]
        path = [np.array(pts[0])]
        for p_ in pts[1:]:
            path.append(path[-1] + ((p_ - path[-1] + R / 2) % R - R / 2))
        return path
    stroke(part(0.024), THICK * 1.9, (255, 255, 255))       # белый ореол (короткий)
    stroke(part(0.070), THICK, VINE_COL)                    # верхняя лоза (длиннее ореола: без щербин)

# цветы поверх
for fx in flowers:
    if fx['shape'] == 'circle':
        circle(fx['x'], fx['y'], fx['r'], fx['col'])
    elif fx['shape'] == 'star':
        poly(star(fx['x'], fx['y'], fx['r'], rng.uniform(0, 3)), fx['col'])
    else:
        poly(ellipse_poly(fx['x'], fx['y'], fx['r'] * 1.15, fx['r'] * 0.8, rng.uniform(0, 3.14)), fx['col'])
        circle(fx['x'], fx['y'], fx['r'] * 0.35, (78, 130, 78))

out = img.resize((R, R), Image.LANCZOS)
out.save(a.out)
print('цветов:', len(flowers), 'листьев:', len(leaves), '→', a.out)
t = Image.new('RGB', (R * 2, R * 2))
for x in (0, 1):
    for y in (0, 1):
        t.paste(out, (x * R, y * R))
t.resize((1400, 1400), Image.LANCZOS).save(a.out.replace('.png', '_проверка_швов2x2.jpg'), quality=90)
