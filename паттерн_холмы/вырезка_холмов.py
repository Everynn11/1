# Вырезка отдельных холмиков из листа Gemini. Два режима: лист уже с прозрачным фоном (берём альфу) или на белом фоне
# (фон = белая область, связанная с краем листа; белые полоски ВНУТРИ холмика не трогаются). Каждый холмик сохраняется в PNG.
# --gapfill R: прозрачные щели шире зазора R px между дугой и заливкой закрываются светло-белой краской (нужно, если Gemini
# оторвал дугу от заливки, а фон потом вычищен: без этого в щелях просвечивает нижний ряд).
#   python -I вырезка_холмов.py --sheet лист_A.png --out элементы --prefix A
import argparse, json, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ap = argparse.ArgumentParser()
ap.add_argument('--sheet', required=True); ap.add_argument('--out', default='элементы'); ap.add_argument('--prefix', default='h')
ap.add_argument('--white', type=int, default=238)      # все каналы >= этого = «белое»
ap.add_argument('--minarea', type=int, default=6000)   # меньше — мусор
ap.add_argument('--gapfill', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
im = Image.open(a.sheet).convert('RGBA'); full = np.array(im); arr = full[..., :3]
has_alpha = (full[..., 3] < 250).mean() > 0.05
if has_alpha:
    A = full[..., 3].astype(float) / 255
    fg = A > 0.05
else:
    white = (arr >= a.white).all(axis=2)
    lab, n = ndi.label(white, structure=np.ones((3, 3)))
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])); border = border[border > 0]
    fg = ~np.isin(lab, border)                              # всё, что не белое-связанное-с-краем
    A = None
fg = ndi.binary_opening(fg, iterations=2)
fg = ndi.binary_closing(fg, iterations=4)
fg = ndi.binary_fill_holes(fg)
lab, n = ndi.label(fg, structure=np.ones((3, 3)))
areas = ndi.sum(fg, lab, range(1, n + 1))
rep = []; k = 0
for i, ar in enumerate(areas, 1):
    if ar < a.minarea: continue
    m = lab == i; ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 6; y0, x0 = max(0, y0 - pad), max(0, x0 - pad); y1, x1 = min(arr.shape[0], y1 + pad), min(arr.shape[1], x1 + pad)
    if has_alpha:
        alpha = np.where(m[y0:y1, x0:x1], A[y0:y1, x0:x1], 0)
    else:
        alpha = ndi.gaussian_filter(m[y0:y1, x0:x1].astype(float), 0.9)
        alpha = np.clip((alpha - 0.35) / 0.4, 0, 1)
    rgb = arr[y0:y1, x0:x1].copy()
    solid = ndi.binary_erosion(alpha > 0.97, iterations=2)            # заведомо внутренние пиксели
    idx = ndi.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    edge = ~solid
    rgb[edge] = rgb[idx[0][edge], idx[1][edge]]                         # у краёв цвет берём изнутри: белой каёмки от фона нет
    if a.gapfill > 0:
        R = a.gapfill; yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; disk = (xx * xx + yy * yy) <= R * R
        pad_ = np.pad(alpha > 0.5, R + 2); cl = ndi.binary_closing(pad_, structure=disk)[R + 2:-(R + 2), R + 2:-(R + 2)]
        cl = ndi.binary_fill_holes(cl); add = cl & (alpha < 0.5)
        rgb[add] = (236, 242, 240); alpha = np.where(add, 1.0, alpha)
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    k += 1; name = f'{a.prefix}{k:02d}.png'; Image.fromarray(rgba).save(os.path.join(a.out, name))
    rep.append(dict(file=name, w=int(x1 - x0), h=int(y1 - y0), area=int(ar)))
json.dump(rep, open(os.path.join(a.out, f'_отчёт_{a.prefix}.json'), 'w', encoding='utf8'), ensure_ascii=False)
print(f'вырезано {k} элементов из {a.sheet}:', [(r['w'], r['h']) for r in rep])
