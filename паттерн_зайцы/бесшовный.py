# Делает «почти бесшовную» картинку паттерна (например, из Gemini) по-настоящему бесшовной.
# Gemini часто продолжает элементы через края, но не точно: на стыке лепестки и стебли разрываются.
# Здесь швы переводятся в середину картинки, Gemini перерисовывает только крест из швов, а мы вклеиваем
# его результат обратно мягкой маской. Всё вне креста остаётся пиксель в пиксель, края картинки
# при этом — бывший центр, то есть естественное продолжение друг друга.
#
#   1) python бесшовный.py сдвиг картинка.jpg           → картинка_сдвиг.jpg  (отдать Gemini на починку креста)
#   2) python бесшовный.py сборка картинка_сдвиг.jpg ответ_gemini.png → картинка_бесшовная.jpg + плитка 2×2
import argparse, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['сдвиг', 'сборка', 'проверка'])
ap.add_argument('files', nargs='+')
ap.add_argument('--out', default='')
ap.add_argument('--band', type=float, default=0.07)     # половина ширины полосы креста, доля стороны картинки
ap.add_argument('--feather', type=float, default=0.025) # мягкость края вклейки, доля стороны
ap.add_argument('--quality', type=int, default=95)
a = ap.parse_args()


def load(p):
    return np.array(Image.open(p).convert('RGB'))


def stem(p, suffix):
    return a.out or os.path.splitext(p)[0] + suffix


def seam_report(img):
    f = img.astype(float)
    inner = (np.abs(f[:, 1:] - f[:, :-1]).mean() + np.abs(f[1:] - f[:-1]).mean()) / 2
    lr = np.abs(f[:, -1] - f[:, 0]).mean()
    tb = np.abs(f[-1] - f[0]).mean()
    return inner, lr, tb


def cross_mask(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bx, by = a.band * W, a.band * H
    m = np.maximum(np.clip(1 - (np.abs(xx - W / 2) - bx) / (a.feather * W) , 0, 1),
                   np.clip(1 - (np.abs(yy - H / 2) - by) / (a.feather * H), 0, 1))
    return m


if a.mode == 'сдвиг':
    src = a.files[0]
    img = load(src)
    H, W, _ = img.shape
    inner, lr, tb = seam_report(img)
    print(f'{os.path.basename(src)}: {W}×{H}; обычный перепад между соседними пикселями {inner:.1f}; '
          f'на стыках: лево-право {lr:.1f}, верх-низ {tb:.1f}')
    out = stem(src, '_сдвиг.jpg')
    Image.fromarray(np.roll(np.roll(img, W // 2, axis=1), H // 2, axis=0)).save(out, quality=a.quality, subsampling=0)
    print('сдвинутая картинка (швы в центре крестом):', out)

elif a.mode in ('сборка', 'проверка'):
    if a.mode == 'проверка':
        img = load(a.files[0])
        inner, lr, tb = seam_report(img)
        print(f'перепад соседних {inner:.1f}; стык лево-право {lr:.1f}, верх-низ {tb:.1f}')
        sys.exit()
    shifted, fixed = load(a.files[0]), Image.open(a.files[1]).convert('RGB')
    H, W, _ = shifted.shape
    if fixed.size != (W, H):
        print(f'ответ Gemini {fixed.size[0]}×{fixed.size[1]} → приводим к {W}×{H}')
        fixed = fixed.resize((W, H), Image.LANCZOS)
    B = np.array(fixed).astype(np.float32)
    A = shifted.astype(np.float32)
    m = cross_mask(H, W)
    outer = m < 0.001                                     # вне креста картинки должны совпадать
    # 1) глобальный сдвиг ответа относительно оригинала (Gemini иногда съезжает на пару пикселей)
    ga, gb = A.mean(axis=2) * outer, B.mean(axis=2) * outer
    F = np.fft.fft2(ga - ga[outer].mean() * outer) * np.conj(np.fft.fft2(gb - gb[outer].mean() * outer))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-6)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = (dy - H if dy > H // 2 else dy), (dx - W if dx > W // 2 else dx)
    print(f'сдвиг ответа относительно оригинала: dx={dx}, dy={dy} px')
    if (dx, dy) != (0, 0) and abs(dx) < 40 and abs(dy) < 40:
        B = np.roll(np.roll(B, dy, axis=0), dx, axis=1)
    # 2) выравниваем цвет ответа по оригиналу вне креста (среднее и разброс по каналам)
    for c in range(3):
        mu_a, sd_a = A[..., c][outer].mean(), A[..., c][outer].std()
        mu_b, sd_b = B[..., c][outer].mean(), B[..., c][outer].std()
        B[..., c] = (B[..., c] - mu_b) * (sd_a / max(sd_b, 1e-3)) + mu_a
    diff_outer = np.abs(A - B)[outer].mean()
    print(f'расхождение ответа с оригиналом вне креста: {diff_outer:.1f} (чем меньше, тем лучше; до ~8 нормально)')
    out_img = np.clip(A * (1 - m[..., None]) + B * m[..., None], 0, 255).astype(np.uint8)
    out = stem(a.files[0], '_бесшовная.jpg').replace('_сдвиг_бесшовная', '_бесшовная')
    Image.fromarray(out_img).save(out, quality=a.quality, subsampling=0)
    inner, lr, tb = seam_report(out_img)
    print(f'сохранено: {out}; перепад соседних {inner:.1f}; края: лево-право {lr:.1f}, верх-низ {tb:.1f}')
    t = np.tile(out_img, (2, 2, 1))
    Image.fromarray(t).resize((1400, 1400), Image.LANCZOS).save(os.path.splitext(out)[0] + '_плитка2x2.jpg', quality=90)
    cx, cy = W // 2, H // 2
    Image.fromarray(out_img[cy - 350:cy + 350, cx - 350:cx + 350]).save(os.path.splitext(out)[0] + '_крест.png')
    print('плитка 2×2 и крупный план креста сохранены рядом')
