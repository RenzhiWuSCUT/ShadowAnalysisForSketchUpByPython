from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import matplotlib.cm as cm


def ensure_dir(dir_path: str) -> None:
    if dir_path and (not os.path.isdir(dir_path)):
        os.makedirs(dir_path, exist_ok=True)


def scalar_to_rgba_u8(
    img: np.ndarray,
    *,
    clamp: Tuple[float, float] = (0.0, 1.0),
    cmap_name: str = "turbo",
) -> np.ndarray:
    a = img.astype(np.float32, copy=False)
    lo = float(clamp[0])
    hi = float(clamp[1])
    if hi <= lo:
        hi = lo + 1e-6
    x = (a - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(x)  # float in 0..1, shape(H,W,4)
    out = (rgba * 255.0 + 0.5).astype(np.uint8)
    return out


def save_rgba_png(path: str, rgba_u8: np.ndarray) -> None:
    # 用matplotlib写png，避免额外依赖
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(path))
    plt.imsave(path, rgba_u8)