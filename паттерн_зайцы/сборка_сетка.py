# Паттерн «заяц на цветущем лугу»: ромбическая сетка (half-drop), бесшовно (тор).
# Никакой случайности: у каждого элемента свой фиксированный размер, позы зайцев идут по схеме.
#
# Узлы сетки (nx столбцов, ny строк, ny чётное; строки чередуются со сдвигом на полшага):
#   A — чётные строки, без сдвига       → зайцы
#   B — нечётные строки, сдвиг на ½     → цветы
#   C — чётные строки, сдвиг на ½       → ветки   (слой выключен, включить: --layers A,B,C)
#   D — нечётные строки, без сдвига     → колосья, птицы, филлеры (--layers A,B,C,D)
# Раппорт 29" = 4350 px @150 dpi, высота 5200 px.
import argparse, math, os
import numpy as np
from PIL import Image, ImageDraw

PX_CM = 150 / 2.54

# Размер элемента на ткани, см. Калибр = корень из площади НЕПРОЗРАЧНЫХ пикселей (масса фигуры),
# поэтому позы одного зайца получаются одного размера, а не по габаритам.
CM = {'заяц': 9.0, 'бабочка': 3.2, 'малиновка': 5.5, 'анемона': 6.5, 'бутон': 5.5,
      'папоротник': 5.5, 'эвкалипт': 5.5, 'ягоды': 5.5, 'дуб': 5.5, 'колос': 4.5, 'филлер': 5.0}

LAYERS = {
    'A': ['заяц_база', 'заяц_нюхает', 'заяц_прыгает', 'заяц_спит', 'заяц_столбиком'],
    'B': ['анемона_1', 'анемона_2', 'бутон_1'],
    'C': ['папоротник_1', 'ягоды_1', 'эвкалипт_1', 'папоротник_2', 'дуб_1'],
    'D': ['колос_1', 'малиновка_1', 'филлер_2', 'малиновка_2', 'филлер_1'],
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
ap.add_argument('--trellis', default='')                # цвет решётки (HEX), напр. D9CFBA; пусто = без решётки
ap.add_argument('--trellis-w', type=float, default=1.8)  # толщина линий решётки, мм
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
    im = im.crop(bb)
    mass = math.sqrt((np.array(im.getchannel('A')) > 40).sum())
    k = cm * PX_CM * S / mass
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
    """Нос спящего зайца: на исходной картинке он на самом правом краю."""
    al = np.array(el.getchannel('A')) > 40
    cols = np.nonzero(al.any(axis=0))[0]
    c = cols[-1] if not flipped else cols[0]
    band = cols[-max(3, len(cols) // 40):] if not flipped else cols[:max(3, len(cols) // 40)]
    ys = np.nonzero(al[:, band].any(axis=1))[0]
    return int(c), int(ys.mean())


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
            el = cache[name].transpose(Image.FLIP_LEFT_RIGHT) if flip else cache[name]
            paste(el, cx, cy)
            if name == 'заяц_спит' and not a.no_butterfly:     # бабочка (в 3/4, смотрит к морде) на носу
                if 'бабочка_2' not in cache:
                    cache['бабочка_2'], _ = load('бабочка_2', CM['бабочка'])
                b = cache['бабочка_2']
                if not flip:
                    b = b.transpose(Image.FLIP_LEFT_RIGHT)    # исходная смотрит вправо, нос справа: зеркалим к морде
                nx_, ny_ = nose(el, flip)
                px, py = cx - el.width / 2 + nx_, cy - el.height / 2 + ny_
                paste(b, px + (-1 if not flip else 1) * 0.10 * b.width, py - 0.45 * b.height)

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
