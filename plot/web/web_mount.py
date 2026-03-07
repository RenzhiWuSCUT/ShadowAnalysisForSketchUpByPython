from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List,Tuple

import numpy as np

from obj.Geo import SketchUpGeo
from util.geo_sampling import GeoSampling


def uv_to_xyz(geo: SketchUpGeo, fi: int, uv: np.ndarray) -> np.ndarray:
    """
    uv: (N,2) 该face局部平面坐标(uu,vv)
    返回: (N,3) 世界坐标
    """
    basis = GeoSampling.face_plane_basis(geo, int(fi))
    p0 = basis.p0.astype(np.float64, copy=False)
    u = basis.u.astype(np.float64, copy=False)
    v = basis.v.astype(np.float64, copy=False)
    uu = uv[:, 0].astype(np.float64, copy=False)
    vv = uv[:, 1].astype(np.float64, copy=False)
    return (p0[None, :] + uu[:, None] * u[None, :] + vv[:, None] * v[None, :]).astype(np.float64)


def _loop_point_indices(geo: SketchUpGeo, loop_idx: int) -> List[int]:
    """
    返回该loop的点索引环（不闭合）
    优先使用 geo.loops[loop_idx].verts
    若为空 再从edges尽力串起来
    """
    lp = geo.loops[int(loop_idx)]

    if hasattr(lp, "verts") and len(lp.verts) >= 2:
        ring = [int(v) for v in lp.verts]
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        return ring

    if len(lp.edges) == 0:
        return []

    ring: List[int] = []
    e0 = geo.edges[int(lp.edges[0])]
    ring.append(int(e0.a))
    ring.append(int(e0.b))

    for ei in lp.edges[1:]:
        e = geo.edges[int(ei)]
        a = int(e.a)
        b = int(e.b)
        last = ring[-1]
        head = ring[0]

        if a == last:
            ring.append(b)
        elif b == last:
            ring.append(a)
        elif a == head:
            ring.insert(0, b)
        elif b == head:
            ring.insert(0, a)
        else:
            continue

    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def loop_xyz(geo: SketchUpGeo, loop_idx: int) -> np.ndarray:
    """
    返回 (N,3) 世界坐标点序列（不闭合）
    顶点列表是 geo.points
    """
    vidx = _loop_point_indices(geo, int(loop_idx))
    if len(vidx) < 2:
        return np.zeros((0, 3), dtype=np.float64)

    pts = np.zeros((len(vidx), 3), dtype=np.float64)
    for i, vi in enumerate(vidx):
        p = geo.points[int(vi)]
        pts[i, 0] = float(p.x)
        pts[i, 1] = float(p.y)
        pts[i, 2] = float(p.z)
    return pts


def prepare_web_dirs(*, temp_web_dir: str) -> tuple[str, str, str]:
    out_dir = os.path.abspath(temp_web_dir)
    tex_dir = os.path.join(out_dir, "textures")
    js_dir = os.path.join(out_dir, "js")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(js_dir, exist_ok=True)
    return out_dir, tex_dir, js_dir


def require_web_deps(*, js_dir: str) -> None:
    three_path = os.path.join(js_dir, "three.min.js")
    orbit_path = os.path.join(js_dir, "OrbitControls.js")
    if not os.path.isfile(three_path):
        raise FileNotFoundError(f"missing: {three_path}  请把 three.min.js 放到 temp_web/js/")
    if not os.path.isfile(orbit_path):
        raise FileNotFoundError(f"missing: {orbit_path}  请把 OrbitControls.js 放到 temp_web/js/ (必须是非ESM版本)")


def _rgba_to_css_hex(rgb01) -> str:
    r = int(np.clip(rgb01[0], 0.0, 1.0) * 255.0 + 0.5)
    g = int(np.clip(rgb01[1], 0.0, 1.0) * 255.0 + 0.5)
    b = int(np.clip(rgb01[2], 0.0, 1.0) * 255.0 + 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_cbar_data(
    *,
    cmap_name: str,
    vis_clamp: Tuple[float, float],
    n_bins: int = 10,
) -> dict:
    import matplotlib.cm as cm

    v0 = float(vis_clamp[0])
    v1 = float(vis_clamp[1])
    if v1 <= v0:
        v1 = v0 + 1e-6

    if n_bins <= 1:
        n_bins = 2

    if (cmap_name or "").lower() == "turbo":
        cmap = cm.get_cmap("turbo")
    else:
        cmap = cm.get_cmap(cmap_name)

    colors: List[str] = []
    values: List[float] = []

    for i in range(n_bins):
        t = 1.0 - i / float(n_bins - 1)
        rgba = cmap(t)
        colors.append(_rgba_to_css_hex(rgba[:3]))
        values.append(v0 + (v1 - v0) * t)

    return {
        "title": "日照百分比",
        "unit": "%",
        "n_bins": int(n_bins),
        "colors": colors,
        "values": values,
        "cmap_name": cmap_name,
    }


def build_web_data(
    *,
    items,
    title: str,
    show_outline: bool,
    quad_opacity: float,
    outline_color: str,
    hole_color: str,
    vis_clamp: Tuple[float, float],
    cmap_name: str,
) -> dict:
    faces = []
    for it in items:
        faces.append(
            {
                "fi": int(it.fi),
                "quad4": np.asarray(it.quad4, dtype=np.float64).tolist(),
                "tex": str(it.tex_rel),
                "outer": np.asarray(it.outer, dtype=np.float64).tolist(),
                "holes": [np.asarray(h, dtype=np.float64).tolist() for h in it.holes],
            }
        )

    return {
        "title": title,
        "faces": faces,
        "show_outline": bool(show_outline),
        "quad_opacity": float(quad_opacity),
        "outline_color": str(outline_color),
        "hole_color": str(hole_color),
        "vis_clamp": [float(vis_clamp[0]), float(vis_clamp[1])],
        "cmap_name": str(cmap_name),
        "cbar": _build_cbar_data(
            cmap_name=cmap_name,
            vis_clamp=vis_clamp,
            n_bins=10,
        ),
    }