from __future__ import annotations

import os
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from obj.Geo import SketchUpGeo
from .web.img_mount import scalar_to_png
from .web.web_mount import (
    uv_to_xyz,
    loop_xyz,
    build_web_data,
    prepare_web_dirs,
    require_web_deps,
)
from .web.web_write import *


@dataclass
class _FaceWebItem:
    fi: int
    quad4: np.ndarray
    tex_rel: str
    outer: np.ndarray
    holes: List[np.ndarray]


class SampleIntensityInterpPreview:
    @staticmethod
    def show_points(
            geo: SketchUpGeo,
            geo4,
            face_to_scalar: Dict[int, np.ndarray],
            *,
            title: str = "采样点强度插值预览 Three.js",
            tex_size: int = 256,
            vis_clamp: Tuple[float, float] = (0.0, 1.0),
            cmap_name: str = "turbo",
            show_outline: bool = True,
            quad_opacity: float = 1.0,
            outline_color: str = "#ffffff",
            hole_color: str = "#ffffff",
            temp_web_dir: str = "web模型",
    ) -> None:
        if tex_size <= 1:
            raise ValueError("tex_size must be >= 2")

        faceBoundUVL = geo4.faceBoundUVL
        face_count = int(faceBoundUVL.shape[0])

        has_valid = hasattr(geo4, "faceBoundValidL")
        if has_valid:
            faceBoundValidL = geo4.faceBoundValidL

        out_dir, tex_dir, js_dir = prepare_web_dirs(temp_web_dir=temp_web_dir)

        write_file(out_dir, "index.html")
        write_file(out_dir, "app.js")
        write_file(out_dir, "style.css")
        write_file(out_dir, "cbar.js")
        write_file(out_dir, "cbar.css")
        require_web_deps(js_dir=js_dir)

        items: List[_FaceWebItem] = []

        for fi, img in face_to_scalar.items():
            fi = int(fi)
            if fi < 0 or fi >= face_count:
                continue
            if has_valid and int(faceBoundValidL[fi]) == 0:
                continue

            umin, umax, vmin, vmax = [float(x) for x in faceBoundUVL[fi]]
            du = umax - umin
            dv = vmax - vmin
            if du <= 0.0:
                du = 1e-6
            if dv <= 0.0:
                dv = 1e-6

            uv4 = np.array(
                [
                    [umin, vmin],
                    [umax, vmin],
                    [umax, vmax],
                    [umin, vmax],
                ],
                dtype=np.float64,
            )
            quad4 = uv_to_xyz(geo, fi, uv4)

            png_name = f"face_{fi:06d}.png"
            png_path = os.path.join(tex_dir, png_name)
            scalar_to_png(img, png_path, vis_clamp=vis_clamp, cmap_name=cmap_name)
            tex_rel = f"textures/{png_name}"

            face = geo.faces[fi]
            outer = loop_xyz(geo, int(face.outer_loop))
            holes_xyz: List[np.ndarray] = []
            for li in getattr(face, "inner_loops", []):
                holes_xyz.append(loop_xyz(geo, int(li)))

            items.append(
                _FaceWebItem(
                    fi=fi,
                    quad4=quad4,
                    tex_rel=tex_rel,
                    outer=outer,
                    holes=holes_xyz,
                )
            )

        if len(items) == 0:
            raise ValueError("no valid faces in face_to_scalar (or faceBoundValidL all invalid)")

        data = build_web_data(
            items=items,
            title=title,
            show_outline=show_outline,
            quad_opacity=quad_opacity,
            outline_color=outline_color,
            hole_color=hole_color,
            vis_clamp=vis_clamp,
            cmap_name=cmap_name,
        )
        write_data_js(out_dir, data)

        url = "file:///" + os.path.join(out_dir, "index.html").replace("\\", "/")
        webbrowser.open(url)