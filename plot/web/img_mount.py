from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import matplotlib.cm as cm
from PIL import Image


def scalar_to_png(
    img: np.ndarray,
    out_png: str,
    *,
    vis_clamp: Tuple[float, float],
    cmap_name: str,
) -> None:
    """
    img: (H,W) float32
    输出RGBA png
    """
    a = img.astype(np.float32, copy=False)
    v0 = float(vis_clamp[0])
    v1 = float(vis_clamp[1])
    if v1 <= v0:
        v1 = v0 + 1e-6

    a = np.clip(a, v0, v1)
    a = (a - v0) / (v1 - v0)
    a = a[::-1, :]
    if (cmap_name or "").lower() == "turbo":
        cmap = cm.get_cmap("turbo")
    else:
        cmap = cm.get_cmap(cmap_name)

    rgba = cmap(a)
    u8 = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    out_parent = os.path.dirname(out_png)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    Image.fromarray(u8, mode="RGBA").save(out_png)