# Белый фон → прозрачность, мягкий край, убирает мелкий мусор.
# Копия эталона из «дальняя полка». Используйте glob=True для растений
# (белое убирается везде, и в просветах между листьями), для зайцев и птиц — по умолчанию.
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi


def cut(src, dst, d0=8, d1=60, glob=False):
    im = np.array(Image.open(src).convert('RGB')).astype(np.float32)
    d = 255 - im.min(axis=2)  # расстояние от белого
    near = d < d1
    lab, n = ndi.label(near)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    bg = np.isin(lab, border)  # белое, связанное с краем кадра
    if glob:
        bg = near
    t = np.clip((d - d0) / (d1 - d0), 0, 1)
    t = t * t * (3 - 2 * t)
    alpha = np.where(bg, t, 1.0)
    # снять белую примесь на мягких краях
    a = np.maximum(alpha, 1e-3)[..., None]
    col = np.where((alpha < 1)[..., None], 255 - (255 - im) / a, im)
    col = np.clip(col, 0, 255)
    out = np.dstack([col, alpha * 255]).astype(np.uint8)
    # убрать мусор: мелкие компоненты альфы
    al = out[..., 3] > 20
    lab2, n2 = ndi.label(al, structure=np.ones((3, 3)))
    sizes = ndi.sum(al, lab2, range(1, n2 + 1))
    keep = np.isin(lab2, [i + 1 for i, s in enumerate(sizes) if s > 400])
    out[..., 3] = np.where(keep, out[..., 3], 0)
    Image.fromarray(out, 'RGBA').save(dst)


if __name__ == '__main__':
    cut(sys.argv[1], sys.argv[2], glob=('--glob' in sys.argv))
