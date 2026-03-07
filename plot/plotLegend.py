# plot/plotLegend.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from obj.Geo import SketchUpGeo, Point3, Loop, Face, Texture, Material


def _meters_to_su_internal_inch(units: str) -> float:
    u = (units or "").strip()
    if u == "Meters":
        model_units_per_m = 1.0
    elif u == "Centimeters":
        model_units_per_m = 100.0
    elif u == "Millimeters":
        model_units_per_m = 1000.0
    elif u == "Feet":
        model_units_per_m = 1.0 / 0.3048
    elif u == "Inches":
        model_units_per_m = 1.0 / 0.0254
    else:
        model_units_per_m = 1.0
    return float(model_units_per_m) / 0.0254


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def _save_discrete_blocks_png(
    out_png: str,
    *,
    n_blocks: int,
    cmap_name: str,
    size_px: Tuple[int, int],
    dpi: int,
) -> None:
    _ensure_dir(out_png)
    W = int(size_px[0])
    H = int(size_px[1])
    if W <= 0 or H <= 0:
        raise ValueError("size_px必须>0")
    if int(n_blocks) <= 0:
        raise ValueError("n_blocks必须>0")

    img = np.zeros((H, W, 4), dtype=np.float32)
    cmap = plt.get_cmap(str(cmap_name))

    xs = (np.arange(int(n_blocks), dtype=np.float32) + 0.5) / float(n_blocks)
    cols = np.stack([cmap(float(x)) for x in xs], axis=0).astype(np.float32)  # (n,4)

    bw = max(W // int(n_blocks), 1)
    for i in range(int(n_blocks)):
        x0 = i * bw
        x1 = W if (i == int(n_blocks) - 1) else (i + 1) * bw
        img[:, x0:x1, :] = cols[i][None, None, :]

    # 直接旋转90度（逆时针），不暴露任何开关
    img = np.rot90(img, k=1).copy()
    img = img[::-1, :, :].copy()
    fig_w = float(img.shape[1]) / float(dpi)
    fig_h = float(img.shape[0]) / float(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_png, transparent=True)
    plt.close(fig)


@dataclass
class PlotLegend:
    @staticmethod
    def attach_legend(
        geo: SketchUpGeo,
        *,
        out_dir: str,
        units: str,
        cmap_name: str = "turbo",
        png_name: str = "legend.png",
        n_blocks: int = 10,
        png_size_px: Tuple[int, int] = (640, 120),
        dpi: int = 200,
        width_m: float = 18.0,
        height_m: float = 3.0,
        offset_m: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> int:
        if len(geo.points) == 0:
            raise ValueError("geo.points为空，无法放置legend")

        out_png = os.path.join(str(out_dir), str(png_name))
        _save_discrete_blocks_png(
            out_png,
            n_blocks=int(n_blocks),
            cmap_name=str(cmap_name),
            size_px=(int(png_size_px[0]), int(png_size_px[1])),
            dpi=int(dpi),
        )

        tex_i = len(geo.textures)
        geo.textures.append(Texture(png=str(out_png)))

        mat_i = len(geo.materials)
        geo.materials.append(Material(name=f"legend_{mat_i:03d}", texture_index=int(tex_i)))

        P = np.array([[p.x, p.y, p.z] for p in geo.points], dtype=np.float64)
        bmin = np.min(P, axis=0)

        s = _meters_to_su_internal_inch(units)
        dx, dy, dz = float(offset_m[0]) * s, float(offset_m[1]) * s, float(offset_m[2]) * s
        Wm = float(width_m) * s
        Hm = float(height_m) * s
        if Wm <= 0.0 or Hm <= 0.0:
            raise ValueError("width_m/height_m必须>0")

        x0 = float(bmin[0] + dx)
        y0 = float(bmin[1] + dy)
        z0 = float(bmin[2] + dz)

        pts = [
            (x0, y0, z0),
            (x0 + Wm, y0, z0),
            (x0 + Wm, y0 + Hm, z0),
            (x0, y0 + Hm, z0),
        ]

        vidx = []
        for x, y, z in pts:
            geo.points.append(Point3(float(x), float(y), float(z)))
            vidx.append(len(geo.points) - 1)

        loop_i = len(geo.loops)
        geo.loops.append(Loop(is_outer=True, edges=[], verts=[int(v) for v in vidx]))

        face_i = len(geo.faces)
        geo.faces.append(
            Face(
                outer_loop=int(loop_i),
                inner_loops=[],
                front_material=int(mat_i),
                back_material=-1,
                n=Point3(0.0, 0.0, 1.0),
            )
        )
        return int(face_i)