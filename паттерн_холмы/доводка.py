"""Доводка готовой плитки до «акварельной» живописности. Бесшовность сохраняется:
шум строится через FFT (периодичен по построению), расстояния считаются на плитке 3x3.
Запуск: python -I доводка.py --in плитка.png --out плитка_доводка.png [--pool 0.14 --wash 0.06 --grain 2.5]
"""
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--fill', default='567783')       # цвет заливки холмика
ap.add_argument('--pool', type=float, default=0.14)   # скопление пигмента у тёмной полосы (доля яркости)
ap.add_argument('--poolw', type=float, default=45.0)  # ширина скопления, px
ap.add_argument('--wash', type=float, default=0.06)   # лёгкие разводы цвета в заливке
ap.add_argument('--bandwash', type=float, default=0.05)  # разводы в тёмных полосах
ap.add_argument('--hue', type=float, default=5.0)     # сдвиг оттенка заливки между разводами (уровни RGB)
ap.add_argument('--grain', type=float, default=2.5)   # мелкое зерно бумаги (уровни)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()

img = np.array(Image.open(a.src).convert('RGB')).astype(np.float32)
H, W = img.shape[:2]
rng = np.random.default_rng(a.seed)


def pnoise(scale, beta=2.0):
    """Периодический шум ~1/f^beta, нормирован до std=1; scale — характерный размер пятна, px."""
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2); f[0, 0] = 1
    amp = (f * scale) ** -beta * np.exp(-((f * scale / 1.0) ** 0) * 0)  # степенной спектр
    amp[f * scale < 0.6] *= 0.3                                           # убираем самые крупные ползущие градиенты
    amp[0, 0] = 0
    z = np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((H, W)))).real
    return (z / z.std()).astype(np.float32)


fc = np.array([int(a.fill[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
lum = img @ np.array([.299, .587, .114], np.float32)
dist_fill = np.linalg.norm(img - fc, axis=2)
w_fill = np.clip(1 - dist_fill / 30, 0, 1)                      # мягкая маска заливки
w_dark = np.clip((95 - lum) / 35, 0, 1) * (1 - w_fill)          # мягкая маска тёмных полос
w_light = np.clip((lum - 150) / 40, 0, 1)                       # светлые линии — почти не трогаем

# расстояние внутри заливки до ближайшего НЕ-заливочного пикселя (на плитке 3x3, чтобы стыки совпали)
m = (w_fill > .5)
m3 = np.tile(m, (3, 3))
d3 = ndi.distance_transform_edt(m3)
d = d3[H:2 * H, W:2 * W].astype(np.float32)
pool = np.exp(-d / a.poolw)                                      # 1 у края, 0 вдали

# 1) заливка: скопление пигмента у краёв + разводы + лёгкий сдвиг оттенка
big = pnoise(260); mid = pnoise(90)
k = 1 - a.pool * pool + a.wash * (0.7 * big + 0.5 * mid)
fillmod = img * k[..., None]
h1 = pnoise(200)
tint = np.stack([-h1, 0.4 * h1, h1], -1) * a.hue                 # тёплее/холоднее по пятнам
fillmod = fillmod + tint * w_fill[..., None]
out = img * (1 - w_fill[..., None]) + fillmod * w_fill[..., None]

# 2) тёмные полосы: разводы и «мазки» вдоль, без изменения края
bn = pnoise(120) * 0.6 + pnoise(35) * 0.4
out = out * (1 + a.bandwash * bn * w_dark)[..., None]

# 3) зерно бумаги по всему, чуть слабее на светлых линиях
g = rng.normal(0, a.grain, (H, W, 1)).astype(np.float32)
g = ndi.gaussian_filter(g, (0.7, 0.7, 0), mode='wrap') * 1.8
out = out + g * (1 - 0.6 * w_light[..., None])

Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)

# проверка шва: разница соседних колонок на краю против внутри
o = np.clip(out, 0, 255)
e = (np.abs(o[:, 0] - o[:, -1]).mean() + np.abs(o[0] - o[-1]).mean()) / 2
i = (np.abs(o[:, 1:] - o[:, :-1]).mean() + np.abs(o[1:] - o[:-1]).mean()) / 2
print(f'шов {e:.2f}  внутри {i:.2f}')
