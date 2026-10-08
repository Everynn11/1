# Паттерн «заяц на цветущем лугу»: ромбическая сетка (half-drop), бесшовно (тор).
# Никакой случайности: у каждого элемента свой фиксированный размер, позы зайцев идут по схеме.
#
# Узлы сетки (nx столбцов, ny строк, ny чётное; строки чередуются со сдвигом на полшага):
#   A — чётные строки, без сдвига       → зайцы
#   B — нечётные строки, сдвиг на ½     → цветы
#   C — чётные строки, сдвиг на ½       → (занято зайцами, места нет)
#   D — нечётные строки, без сдвига     → ветки между цветами (--layers A,B,D)
# Раппорт 29" = 4350 px @150 dpi, высота 5200 px.
import argparse, math, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

PX_CM = 150 / 2.54

# Размер элемента на ткани, см. Калибр = корень из площади НЕПРОЗРАЧНЫХ пикселей (масса фигуры),
# поэтому позы одного зайца получаются одного размера, а не по габаритам.
CM = {'заяц': 9.0, 'бабочка': 3.2, 'малиновка': 2.6, 'синица': 2.6, 'воробей': 2.6, 'птица': 2.6, 'анемона': 6.5, 'бутон': 5.5,
      'папоротник': 5.5, 'эвкалипт': 5.5, 'ягоды': 5.5, 'дуб': 5.5, 'колос': 4.5, 'филлер': 5.0}

LAYERS = {
    'A': ['заяц_база', 'заяц_нюхает', 'заяц_прыгает', 'заяц_спит', 'заяц_столбиком'],
    'B': ['анемона_1', 'анемона_2', 'бутон_1'],
    'C': [],      # между зайцами в их рядах: места нет (зазор 2–7 см), слой не используется
    'D': ['папоротник_1', 'ягоды_1', 'эвкалипт_1', 'папоротник_2', 'дуб_1'],   # между цветами в нечётных рядах
}

ap = argparse.ArgumentParser()
ap.add_argument('--shelf', required=True)
ap.add_argument('--out', default='раппорт_сетка.jpg')
ap.add_argument('--width', type=int, default=4350)
ap.add_argument('--height', type=int, default=5200)
ap.add_argument('--nx', type=int, default=4)            # зайцев в строке
ap.add_argument('--ny', type=int, default=6)            # строк всего (чётное)
ap.add_argument('--layers', default='A,B')
ap.add_argument('--bg', default='ECE5D8')
ap.add_argument('--scale', type=float, default=1.0)     # 0.25 = черновик
ap.add_argument('--quality', type=int, default=95)
ap.add_argument('--size', type=float, default=1.07)     # общий множитель размеров элементов (шаг сетки не меняется)
ap.add_argument('--trellis', default='')                # цвет решётки (HEX), напр. D9CFBA; пусто = без решётки
ap.add_argument('--trellis-w', type=float, default=1.8)  # толщина линий решётки, мм
ap.add_argument('--land-dx', type=float, default=0.0)    # смещение туловища бабочки от кончика носа (+ от морды), доля её ширины
ap.add_argument('--tilt', type=float, default=14)        # наклон бабочки по часовой стрелке, градусов
ap.add_argument('--land-gap', type=float, default=0.0)   # поднять бабочку над точкой касания, доля её высоты
ap.add_argument('--no-birds', action='store_true')         # птицы на изгибе бутонов
ap.add_argument('--no-butterfly', action='store_true')  # бабочка на носу спящего зайца
ap.add_argument('--check', action='store_true')
a = ap.parse_args()
assert a.ny % 2 == 0, '--ny должно быть чётным (иначе сетка не сойдётся на шве)'

S = a.scale
W, H = int(a.width * S), int(a.height * S)
canvas = Image.new('RGB', (W, H), tuple(int(a.bg[i:i+2], 16) for i in (0, 2, 4)))
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
    for ddx in (-W, 0, W):                               # вылезшее за край возвращается с другой стороны
        for ddy in (-H, 0, H):
            xx, yy = x + ddx, y + ddy
            if xx < W and yy < H and xx + el.width > 0 and yy + el.height > 0:
                canvas.paste(rgb, (xx, yy), al)


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


if a.trellis:                                            # ромбическая решётка сквозь узлы A и B, под элементами
    dr = ImageDraw.Draw(canvas)
    col = tuple(int(a.trellis[i:i+2], 16) for i in (0, 2, 4))
    lw = max(1, round(a.trellis_w / 10 * PX_CM * S))
    L = W + H
    for i in range(-a.nx, 2 * a.nx):
        for jr in range(-a.ny, 2 * a.ny):
            x0, y0 = ox + i * dx, oy + 2 * jr * dy
            for sgn in (1, -1):
                ux, uy = sgn * dx / 2, dy
                n = math.hypot(ux, uy)
                dr.line([(x0 - ux / n * L, y0 - uy / n * L), (x0 + ux / n * L, y0 + uy / n * L)], fill=col, width=lw)

cache = {}
sizes = []
bird_n = 0
BIRDS = sorted(os.path.splitext(f)[0] for f in os.listdir(a.shelf)
               if f.endswith('.png') and prefix_of(f) in ('малиновка', 'синица', 'воробей', 'птица'))
for layer in a.layers.split(','):
    names = LAYERS[layer]
    for jr in range(a.ny // 2):
        for i in range(a.nx):
            name = names[(i + 2 * jr) % len(names)] if layer == 'A' else names[(i + jr) % len(names)]
            if name not in cache:
                cache[name], k = load(name, CM[prefix_of(name)])
                sizes.append((name, cache[name].size, k))
            cx = ox + i * dx + (0.5 * dx if layer in 'BC' else 0)
            cy = oy + (2 * jr + (1 if layer in 'BD' else 0)) * dy
            flip = (i + jr) % 2 == 1
            if name == 'заяц_спит' and layer == 'A':           # нос (с бабочкой) смотрит в сторону большего просвета
                half = {}
                for side, di in (('l', -1), ('r', 1)):
                    nb = names[((i + di) % a.nx + 2 * jr) % len(names)]
                    if nb not in cache:
                        cache[nb], k = load(nb, CM[prefix_of(nb)])
                        sizes.append((nb, cache[nb].size, k))
                    half[side] = cache[nb].width / 2
                w_me = cache[name].width / 2
                flip = (dx - w_me - half['l']) > (dx - w_me - half['r'])
            if name == 'заяц_прыгает' and layer == 'A' and not a.no_birds and BIRDS:
                # исходно прыгает вправо, задние лапы слева: лапы разворачиваем от бутонов с птицами
                bl = sum(LAYERS['B'][(((i - 1) % a.nx) + jb) % 3].startswith('бутон') for jb in (jr - 1, jr))
                br = sum(LAYERS['B'][(i + jb) % 3].startswith('бутон') for jb in (jr - 1, jr))
                if bl != br:
                    flip = bl > br
            el = cache[name].transpose(Image.FLIP_LEFT_RIGHT) if flip else cache[name]
            if name.startswith('бутон') and not a.no_birds and BIRDS:   # птица садится на верхний изгиб, стебель закрывает лапки
                bn = BIRDS[bird_n % len(BIRDS)]
                if bn not in cache:
                    cache[bn], k = load(bn, CM[prefix_of(bn)])
                    sizes.append((bn, cache[bn].size, k))
                bird = cache[bn].transpose(Image.FLIP_LEFT_RIGHT) if bird_n % 4 in (1, 2) else cache[bn]
                ax, ay, t = apex(el)
                fx, fy = foot_anchor(bird)
                bx = cx - el.width / 2 + ax - fx
                by = cy - el.height / 2 + ay + 0.5 * t - fy
                paste(bird, bx + bird.width / 2, by + bird.height / 2)
                bird_n += 1
            paste(el, cx, cy)
            if name == 'заяц_спит' and not a.no_butterfly:     # бабочка (в 3/4, смотрит к морде) на носу
                if 'бабочка_2' not in cache:
                    cache['бабочка_2'], _ = load('бабочка_2', CM['бабочка'])
                # исходная бабочка смотрит вправо; голову всегда к морде; наклон по часовой стрелке (у зеркального зайца зеркально)
                b, body = make_butterfly(cache['бабочка_2'], head_left=not flip, angle=(-a.tilt if not flip else a.tilt))
                tx, ty = nose(el, flip)
                lx, ly = land(el, b, body, tx, -1 if flip else 1)
                paste(b, cx - el.width / 2 + lx + b.width / 2, cy - el.height / 2 + ly + b.height / 2)

print('размеры на ткани (см):')
for n, (w, h), k in sizes:
    print(f'  {n:18s} {w / PX_CM / S:5.1f} × {h / PX_CM / S:5.1f}   (исходник ×{k / S:.2f})')
print(f'шаг сетки: {dx / PX_CM / S:.1f} см по горизонтали, {dy / PX_CM / S:.1f} см по вертикали')
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
