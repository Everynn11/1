# Паттерн «заяц на цветущем лугу»: шахматная сетка, бесшовно (тор).
# У каждого элемента свой фиксированный размер, позы и растения расставлены по схеме без повторов рядом.
#
# Узлы: nx столбцов × ny строк (оба чётные). Узел (r, j): если (r + j) чётно — заяц, иначе растение
# (дуб, ягоды, бутон с птицей на изгибе, анемона). В каждой строке зайцы и растения чередуются,
# так что ни горизонтальных, ни вертикальных полос нет, зайцы стоят по диагоналям.
# Дальше в пустоты ставятся средние элементы (соцветия из 2–3 цветков и веточки из листьев),
# потом мелкие филлеры (отдельные листочки/цветочки), всё равномерно, по самым большим дыркам.
# Раппорт 29" = 4350 px @150 dpi, высота 5200 px.
import argparse, math, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

PX_CM = 150 / 2.54

# Размер элемента на ткани, см. Калибр = корень из площади НЕПРОЗРАЧНЫХ пикселей (масса фигуры),
# поэтому позы одного зайца получаются одного размера, а не по габаритам.
CM = {'заяц': 9.0, 'бабочка': 2.67, 'малиновка': 3.9, 'синица': 3.9, 'воробей': 3.9, 'птица': 3.9, 'анемона': 6.5, 'бутон': 5.5,
      'папоротник': 5.5, 'эвкалипт': 5.5, 'ягоды': 5.5, 'дуб': 5.5, 'колос': 4.5, 'филлер': 5.0}

LAYERS = {
    'A': ['заяц_база', 'заяц_нюхает', 'заяц_прыгает', 'заяц_спит', 'заяц_столбиком'],
    'B': ['дуб_1', 'ягоды_1', 'бутон_1', 'анемона_2'],   # чередование в нечётных рядах; бутон в ряду ровно один → одна птица на ряд
}
FILLER_SHEETS = ['филлер_1', 'филлер_2']                  # листы режутся на отдельные листочки и цветочки

ap = argparse.ArgumentParser()
ap.add_argument('--shelf', required=True)
ap.add_argument('--out', default='раппорт_сетка.jpg')
ap.add_argument('--width', type=int, default=4350)
ap.add_argument('--height', type=int, default=6000)
ap.add_argument('--nx', type=int, default=4)            # узлов в строке (чётное)
ap.add_argument('--ny', type=int, default=6)            # строк (чётное)
ap.add_argument('--bg', default='ECE5D8')
ap.add_argument('--scale', type=float, default=1.0)     # 0.25 = черновик
ap.add_argument('--quality', type=int, default=95)
ap.add_argument('--size', type=float, default=1.07)     # общий множитель размеров элементов (шаг сетки не меняется)
ap.add_argument('--land-dx', type=float, default=0.0)    # смещение туловища бабочки от кончика носа (+ от морды), доля её ширины
ap.add_argument('--tilt', type=float, default=14)        # наклон бабочки по часовой стрелке, градусов
ap.add_argument('--land-gap', type=float, default=0.0)   # поднять бабочку над точкой касания, доля её высоты
ap.add_argument('--bud-dy', type=float, default=2.2)       # на сколько см опустить бутон с птицей
ap.add_argument('--no-fillers', action='store_true')
ap.add_argument('--mid-cm', type=float, default=5.0)       # размер средних элементов (соцветий и веточек), см
ap.add_argument('--mid-count', type=int, default=24)
ap.add_argument('--fill-cm', type=float, default=3.3)     # самая крупная деталь филлера, см (меньше бабочки)
ap.add_argument('--fill-gap', type=float, default=0.9)    # просвет вокруг филлера, см
ap.add_argument('--fill-max', type=int, default=1000)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--no-birds', action='store_true')         # птицы на изгибе бутонов
ap.add_argument('--no-butterfly', action='store_true')  # бабочка на носу спящего зайца
ap.add_argument('--check', action='store_true')
a = ap.parse_args()
assert a.ny % 2 == 0 and a.nx % 2 == 0, '--nx и --ny должны быть чётными (иначе шахматка не сойдётся на шве)'

S = a.scale
W, H = int(a.width * S), int(a.height * S)
canvas = Image.new('RGB', (W, H), tuple(int(a.bg[i:i+2], 16) for i in (0, 2, 4)))
occ_alpha = Image.new('L', (W, H), 0)                    # где уже что-то нарисовано (для филлеров)
dx, dy = W / a.nx, H / a.ny
ox, oy = 0.5 * dx, 0.5 * dy                              # сетка сдвинута, чтобы зайцы не лежали на шве


def prefix_of(name):
    return next((p for p in CM if name.startswith(p)), None)


def load(name, cm):
    im = Image.open(os.path.join(a.shelf, name + '.png')).convert('RGBA')
    bb = im.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox()
    im = strip_perch(im.crop(bb), name)
    mass = math.sqrt((np.array(im.getchannel('A')) > 40).sum())
    k = cm * a.size * PX_CM * S / mass
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS), k


def paste(el, cx, cy):
    x, y = int(cx - el.width / 2), int(cy - el.height / 2)
    rgb, al = el.convert('RGB'), el.getchannel('A')
    hard = al.point(lambda v: 255 if v > 40 else 0)
    for ddx in (-W, 0, W):                               # вылезшее за край возвращается с другой стороны
        for ddy in (-H, 0, H):
            xx, yy = x + ddx, y + ddy
            if xx < W and yy < H and xx + el.width > 0 and yy + el.height > 0:
                canvas.paste(rgb, (xx, yy), al)
                occ_alpha.paste(255, (xx, yy, xx + el.width, yy + el.height), hard)


def nose(el, flipped):
    """Кончик носа спящего зайца: на исходной картинке он на самом правом краю.
    Возвращает x кончика и y самой верхней точки носа в самом краю."""
    al = np.array(el.getchannel('A')) > 40
    cols = np.nonzero(al.any(axis=0))[0]
    n = max(3, len(cols) // 60)
    band = cols[:n] if flipped else cols[-n:]
    tip = cols[0] if flipped else cols[-1]
    ys = np.nonzero(al[:, band].any(axis=1))[0]
    return int(tip), int(ys.min())


# Птицы нарисованы вместе с веточкой. Здесь ветка и листочки срезаются (границы — доли габарита),
# чтобы птица села на изгиб стебля. Новые птицы без веточки (файлы синица_*, воробей_*, птица_*) не режутся.
STRIP = {  # имя: (низ, [(x0, y0, x1, y1) области на удаление], удалять_зелёное_слева)
    'малиновка_1': (0.795, [(0.78, 0.62, 1.0, 1.0)], None),
    'малиновка_2': (0.795, [(0.86, 0.66, 1.0, 1.0)], (0.27, 0.50)),
}


def strip_perch(im, name):
    if name not in STRIP:
        return im
    ybot, boxes, green = STRIP[name]
    arr = np.array(im)
    h, w = arr.shape[:2]
    arr[int(ybot * h):, :, 3] = 0
    for x0, y0, x1, y1 in boxes:
        arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), 3] = 0
    if green:                                    # зелёные листочки слева от груди (у малиновки_2)
        gx, gy = green
        reg = arr[int(gy * h):, :int(gx * w)]
        mask = reg[..., 1].astype(int) > reg[..., 0].astype(int) + 5
        reg[..., 3][mask] = 0
    lab, n = ndi.label(arr[..., 3] > 20, structure=np.ones((3, 3)))
    if n > 1:                                    # остаётся только сама птица
        sizes = ndi.sum(arr[..., 3] > 20, lab, range(1, n + 1))
        arr[..., 3][lab != (np.argmax(sizes) + 1)] = 0
    out = Image.fromarray(arr)
    return out.crop(out.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox())


def apex(el):
    """Верхняя точка изгиба бутона: x, верхний y и толщина стебля в этой точке."""
    al = np.array(el.getchannel('A')) > 40
    rows = np.nonzero(al.any(axis=1))[0]
    r0 = rows[0]
    xs = np.nonzero(al[r0:r0 + 3].any(axis=0))[0]
    ax = int(xs.mean())
    col = al[r0:, ax]
    t = int(np.argmin(col)) if (~col).any() else len(col)
    return ax, int(r0), t


def foot_anchor(bird):
    """Низ птицы: x центра лапок и нижняя строка."""
    al = np.array(bird.getchannel('A')) > 40
    rows = np.nonzero(al.any(axis=1))[0]
    r1 = rows[-1]
    xs = np.nonzero(al[max(0, r1 - 4):r1 + 1].any(axis=0))[0]
    return int(xs.mean()), int(r1)


# туловище бабочки (грудь и брюшко) в исходной ориентации (голова справа), доли габарита
BODY = [(0.68, 0.63), (0.60, 0.74), (0.52, 0.90)]


def make_butterfly(base, head_left, angle):
    """Бабочка с маской туловища; поворот угол > 0 против часовой стрелки (PIL)."""
    w, h = base.size
    m = Image.new('L', base.size, 0)
    d = ImageDraw.Draw(m)
    pts = [(x * w, y * h) for x, y in BODY]
    d.line(pts, fill=255, width=int(0.08 * w))
    for px, py in pts:
        r = 0.04 * w
        d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    b = base
    if head_left:
        b, m = b.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
    b = b.rotate(angle, resample=Image.BICUBIC, expand=True)
    m = m.rotate(angle, resample=Image.BILINEAR, expand=True)
    return b, np.array(m) > 128


def land(el, b, body, tip_x, side):
    """Опускает бабочку на нос зайца до первого касания ТУЛОВИЩЕМ («прислонить, едва касаясь»).
    Туловище по x стоит у кончика носа tip_x; side=+1 нос справа, -1 слева. Возвращает левый верхний угол b."""
    ea = np.array(el.getchannel('A')) > 40
    cxm = np.nonzero(body)[1].mean()
    bx = int(tip_x + side * a.land_dx * b.width - cxm)
    y = -b.height - 5
    while y < el.height:
        x0, x1 = max(bx, 0), min(bx + b.width, el.width)
        y0, y1 = max(y, 0), min(y + b.height, el.height)
        if x1 > x0 and y1 > y0 and (ea[y0:y1, x0:x1] & body[y0 - y:y1 - y, x0 - bx:x1 - bx]).any():
            return bx, y
        y += 1
    return bx, y


cache = {}
sizes = []
import random
rnd = random.Random(a.seed)
BIRDS = sorted(os.path.splitext(f)[0] for f in os.listdir(a.shelf)
               if f.endswith('.png') and prefix_of(f) in ('малиновка', 'синица', 'воробей', 'птица'))
birds_on = bool(BIRDS) and not a.no_birds


def get(name):
    if name not in cache:
        cache[name], k = load(name, CM[prefix_of(name)])
        sizes.append((name, cache[name].size, k))
    return cache[name]


def bud(n):
    return n.startswith('бутон')


# ---------- узлы шахматки: (r + j) чётно → заяц, нечётно → растение ----------
nodes = {(r, j): ((r + j) % 2 == 0) for r in range(a.ny) for j in range(a.nx)}


def pos(r, j):
    return ox + j * dx, oy + r * dy


DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))              # ближайшие по диагонали
TIERS = [DIAG + ((2, 0), (-2, 0), (0, 2), (0, -2)),      # от строгого к мягкому: пробуем строгий, если не решается — мягче
         DIAG + ((2, 0), (-2, 0)),
         DIAG + ((0, 2), (0, -2)),
         DIAG]


def assign(kind_nodes, names, row_unique=(), col_unique=(), tiers=None):
    """Раздаёт имена узлам так, чтобы соседи (диагональ, по вертикали через строку, в строке) не совпадали.
    Строгость снижается, если иначе не решается. На каждом уровне — серия случайных попыток,
    берётся решение с самым ровным числом каждого типа (разброс ≤ 1, если получается)."""
    order = sorted(kind_nodes)
    cap = -(-len(order) // len(names)) + 1                  # одного типа не больше (поровну + 1)
    tiers = tiers or TIERS
    for tier, offs in enumerate(tiers):
        best = None
        for attempt in range(80):
            res = {}
            order = sorted(kind_nodes)
            rnd.shuffle(order)

            def rec(k):
                if k == len(order):
                    return True
                n = order[k]
                used = {res[((n[0] + dr) % a.ny, (n[1] + dj) % a.nx)] for dr, dj in offs if abs(dr) == 1
                        and ((n[0] + dr) % a.ny, (n[1] + dj) % a.nx) in res}          # диагональные соседи
                used_far = {res[((n[0] + dr) % a.ny, (n[1] + dj) % a.nx)] for dr, dj in offs if abs(dr) != 1
                            and ((n[0] + dr) % a.ny, (n[1] + dj) % a.nx) in res}      # через строку и в строке
                cnt = {nm: sum(1 for v in res.values() if v == nm) for nm in names}
                for nm in sorted(names, key=lambda x: (cnt[x], rnd.random())):
                    if nm in row_unique and any(res.get((n[0], jj)) == nm for jj in range(a.nx)):
                        continue                                    # такой элемент в строке не больше одного
                    if nm not in used and (nm not in used_far or nm in row_unique) and cnt[nm] < cap:
                        res[n] = nm
                        if rec(k + 1):
                            return True
                        del res[n]
                return False
            if not rec(0):
                break                                       # на этом уровне решения нет совсем
            cs = [sum(1 for v in res.values() if v == nm) for nm in names]
            spread = max(cs) - min(cs)
            same_col = sum(1 for nm in col_unique for m1 in res for m2 in res    # пары одинаковых «редких» элементов в одной колонке
                           if m1 < m2 and res[m1] == nm == res[m2] and m1[1] == m2[1])
            score = (spread > 1, same_col, spread)
            if best is None or score < best[0]:
                best = (score, dict(res), cs)
            if score == (False, 0, 0) or (attempt >= 60 and best[0][0] is False and best[0][1] == 0):
                break
        if best:
            print(f'расстановка {names[0].split("_")[0]}…: строгость {tier + 1} из {len(tiers)}, по типам {best[2]}, одинаковых в колонке: {best[0][1]}')
            return best[1]
    raise AssertionError('не удалось расставить')


TIERS_H = [TIERS[0], DIAG + ((0, 2), (0, -2)), DIAG + ((2, 0), (-2, 0)), DIAG]      # зайцам важнее не повторять позу в строке
hare_at = assign([n for n, h in nodes.items() if h], LAYERS['A'], col_unique=tuple(LAYERS['A']), tiers=TIERS_H)
plant_at = assign([n for n, h in nodes.items() if not h], LAYERS['B'], row_unique=('бутон_1',), col_unique=tuple(LAYERS['B']))
for nm in set(hare_at.values()) | set(plant_at.values()):
    get(nm)

# 1. зайцы
for (r, j), name in hare_at.items():
    cx, cy = pos(r, j)
    flip = ((r + j) // 2) % 2 == 1                        # повторы по строке и по вертикали зеркальны друг другу
    nl, nr = plant_at[(r, (j - 1) % a.nx)], plant_at[(r, (j + 1) % a.nx)]
    if name == 'заяц_спит':                               # нос (с бабочкой) смотрит в сторону большего просвета
        flip = cache[nl].width < cache[nr].width
    if name == 'заяц_прыгает' and birds_on and bud(nl) != bud(nr):
        flip = bud(nl)                                    # задние лапы разворачиваем от бутона с птицей
    el = cache[name].transpose(Image.FLIP_LEFT_RIGHT) if flip else cache[name]
    paste(el, cx, cy)
    if name == 'заяц_спит' and not a.no_butterfly:        # бабочка (в 3/4, смотрит к морде) на носу
        b, body = make_butterfly(get('бабочка_2'), head_left=not flip, angle=(-a.tilt if not flip else a.tilt))
        tx, ty = nose(el, flip)
        lx, ly = land(el, b, body, tx, -1 if flip else 1)
        paste(b, cx - el.width / 2 + lx + b.width / 2, cy - el.height / 2 + ly + b.height / 2)

# 2. растения: дуб, ягоды, бутон (с птицей), анемона
bird_n = 0
for (r, j) in sorted(plant_at):
    name = plant_at[(r, j)]
    cx, cy = pos(r, j)
    el = cache[name].transpose(Image.FLIP_LEFT_RIGHT) if ((r + j) // 2) % 2 else cache[name]
    if bud(name) and birds_on:                            # птица садится на верхний изгиб, стебель закрывает лапки
        cy += a.bud_dy * PX_CM * S
        bird = get(BIRDS[bird_n % len(BIRDS)])
        bird = bird.transpose(Image.FLIP_LEFT_RIGHT) if bird_n % 4 in (1, 2) else bird
        ax, ay, t = apex(el)
        fx, fy = foot_anchor(bird)
        bx = cx - el.width / 2 + ax - fx
        by = cy - el.height / 2 + ay + 0.5 * t - fy
        paste(bird, bx + bird.width / 2, by + bird.height / 2)
        bird_n += 1
    paste(el, cx, cy)


# ---------- филлеры и средние элементы ----------
def split_pieces(name, longest_cm):
    im = Image.open(os.path.join(a.shelf, name + '.png')).convert('RGBA')
    arr = np.array(im)
    al = arr[..., 3] > 40
    lab, n = ndi.label(al, structure=np.ones((3, 3)))
    areas = ndi.sum(al, lab, range(1, n + 1))
    out = []
    for k, ar in enumerate(areas, 1):
        if ar < 0.002 * al.size:
            continue
        m = lab == k
        ys, xs = np.nonzero(m)
        pc = arr.copy()
        pc[..., 3] = np.where(m, arr[..., 3], 0)
        out.append(Image.fromarray(pc).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    big = max(max(p.size) for p in out)
    k = longest_cm * PX_CM * S / big                      # самая крупная деталь листа = longest_cm, остальные пропорционально
    return [p.resize((max(1, int(p.width * k)), max(1, int(p.height * k))), Image.LANCZOS) for p in out]


def orient_up(pc):
    """Деталь на стебельке/листик → головой вверх; возвращает (картинка, x, y основания). Основание — узкий конец."""
    al = np.array(pc.getchannel('A')) > 40
    ys, xs = np.nonzero(al)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    wv, vv = np.linalg.eigh(np.cov((pts - c).T))
    ax = vv[:, 1]
    t, perp = (pts - c) @ ax, (pts - c) @ np.array([-ax[1], ax[0]])
    lo, hi = t.min(), t.max()
    w_lo = perp[t < lo + 0.15 * (hi - lo)].std() if (t < lo + 0.15 * (hi - lo)).sum() > 3 else 1e9
    w_hi = perp[t > hi - 0.15 * (hi - lo)].std() if (t > hi - 0.15 * (hi - lo)).sum() > 3 else 1e9
    head = ax if w_lo < w_hi else -ax                      # голова — на широком конце
    cur = math.degrees(math.atan2(-head[1], head[0]))
    up = pc.rotate(90 - cur, expand=True, resample=Image.BICUBIC)
    ua = np.array(up.getchannel('A')) > 40
    rows = np.nonzero(ua.any(axis=1))[0]
    bx = int(np.nonzero(ua[max(0, rows[-1] - 3):rows[-1] + 1].any(axis=0))[0].mean())
    return up, bx, int(rows[-1])


def fan(pieces, angles):
    """Веерок: детали сходятся основаниями в одной точке и расходятся под углами (градусы, против часовой)."""
    ups = [orient_up(p) for p in pieces]
    R = int(max(max(u.size) for u, _, _ in ups) * 1.6)
    big = Image.new('RGBA', (2 * R, 2 * R), (0, 0, 0, 0))
    for (u, bx, by), ang in zip(ups, angles):
        layer = Image.new('RGBA', (2 * R, 2 * R), (0, 0, 0, 0))
        layer.paste(u, (R - bx, R - by), u)
        big = Image.alpha_composite(big, layer.rotate(ang, center=(R, R), resample=Image.BICUBIC))
    return big.crop(big.getchannel('A').point(lambda v: 255 if v > 10 else 0).getbbox())


DF = 10
MWc, MHc = -(-W // DF), -(-H // DF)


def fill_holes(make, count, margin_cm, label):
    """make() → картинка (RGBA) или None; ставит по очереди в самые большие пустоты."""
    n = 0
    margin = margin_cm * PX_CM * S / DF
    while n < count:
        occ = np.array(occ_alpha.resize((MWc, MHc), Image.BOX)) > 8
        dist = ndi.distance_transform_edt(~np.tile(occ, (3, 3)))[MHc:2 * MHc, MWc:2 * MWc]
        cyc, cxc = np.unravel_index(np.argmax(dist), dist.shape)
        best = dist[cyc, cxc]
        pc = make(best, margin)
        if pc is None:
            break
        paste(pc, cxc * DF + DF / 2, cyc * DF + DF / 2)
        n += 1
    print(f'{label}: {n}')
    return n


mid_n = 0
if not a.no_fillers:
    flowers = split_pieces('филлер_2', a.mid_cm * 0.8)     # цветочки на стебельках → соцветия
    leaves = split_pieces('филлер_1', a.mid_cm * 0.8)      # листочки → веточки

    def make_mid(best, margin):
        for _ in range(12):
            if mid_n % 2 == 0:                             # соцветие из 2–3 цветков, смотрит вверх с небольшим наклоном
                k = rnd.choice((2, 3))
                ps = rnd.sample(flowers, k)
                angs = {2: (-14, 14), 3: (-26, 0, 26)}[k]
                pc = fan(ps, angs).rotate(rnd.uniform(-35, 35), resample=Image.BICUBIC, expand=True)
            else:                                          # листовая веточка из 3–4 листьев, любой ориентации
                k = rnd.choice((3, 4))
                ps = rnd.sample(leaves, k)
                angs = {3: (-30, 0, 30), 4: (-40, -14, 14, 40)}[k]
                pc = fan(ps, angs).rotate(rnd.uniform(0, 360), resample=Image.BICUBIC, expand=True)
            if 0.5 * math.hypot(*pc.size) / DF + margin <= best:
                return pc
        return None

    def make_mid_counted(best, margin):
        global mid_n
        pc = make_mid(best, margin)
        if pc is not None:
            mid_n += 1
        return pc
    fill_holes(make_mid_counted, a.mid_count, a.fill_gap * 1.2, 'средних элементов')

    pool = split_pieces('филлер_1', a.fill_cm) + split_pieces('филлер_2', a.fill_cm)
    print(f'деталей филлера: {len(pool)}, крупнейшая {max(max(p.size) for p in pool) / PX_CM / S:.1f} см')

    def make_fill(best, margin):
        cand = [p for p in pool if 0.5 * math.hypot(*p.size) / DF + margin <= best]
        if not cand:
            return None
        return rnd.choice(cand).rotate(rnd.uniform(0, 360), resample=Image.BICUBIC, expand=True)
    fill_holes(make_fill, a.fill_max, a.fill_gap, 'филлеров')

print('размеры на ткани (см):')
for n, (w, h), k in sizes:
    print(f'  {n:18s} {w / PX_CM / S:5.1f} × {h / PX_CM / S:5.1f}   (исходник ×{k / S:.2f})')
print(f'шаг узлов: {dx / PX_CM / S:.1f} см по строке, {dy / PX_CM / S:.1f} см между строками; раппорт {a.width}×{a.height}')
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
    print('плитка:', chk)
