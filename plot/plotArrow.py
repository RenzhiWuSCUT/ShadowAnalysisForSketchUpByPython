# plot/plotArrow.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from obj.Geo import SketchUpGeo, Point3, Edge


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


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v * 0.0
    return v / n


def _add_pt(geo: SketchUpGeo, p: np.ndarray) -> int:
    geo.points.append(Point3(float(p[0]), float(p[1]), float(p[2])))
    return len(geo.points) - 1


def _add_edge(geo: SketchUpGeo, a: int, b: int) -> int:
    geo.edges.append(Edge(a=int(a), b=int(b)))
    return len(geo.edges) - 1


def _attach_one_arrow(
    geo: SketchUpGeo,
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    L: float,
    head_len_ratio: float,
    head_radius_ratio: float,
    out_edge_idx: list[int],
) -> None:
    d = _normalize(direction.astype(np.float64, copy=False))
    if float(np.linalg.norm(d)) <= 1e-12:
        return

    head_len = max(L * float(head_len_ratio), 1e-6)
    head_len = min(head_len, L * 0.45)
    head_r = max(L * float(head_radius_ratio), 1e-6)

    tip = origin + d * L
    base_center = tip - d * head_len

    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.linalg.norm(np.cross(d, tmp))) <= 1e-6:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(d, tmp))
    v = _normalize(np.cross(d, u))

    b0 = base_center + head_r * (u + v)
    b1 = base_center + head_r * (u - v)
    b2 = base_center + head_r * (-u - v)
    b3 = base_center + head_r * (-u + v)

    i0 = _add_pt(geo, origin)
    ib = _add_pt(geo, base_center)
    it = _add_pt(geo, tip)
    i_b0 = _add_pt(geo, b0)
    i_b1 = _add_pt(geo, b1)
    i_b2 = _add_pt(geo, b2)
    i_b3 = _add_pt(geo, b3)

    out_edge_idx.append(_add_edge(geo, i0, ib))
    out_edge_idx.append(_add_edge(geo, it, i_b0))
    out_edge_idx.append(_add_edge(geo, it, i_b1))
    out_edge_idx.append(_add_edge(geo, it, i_b2))
    out_edge_idx.append(_add_edge(geo, it, i_b3))


@dataclass
class PlotArrow:
    @staticmethod
    def attach_sun_dirs(
        geo: SketchUpGeo,
        *,
        units: str,
        sun_dirs: np.ndarray,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        length_m: float = 50.0,
        head_len_ratio: float = 0.1,
        head_radius_ratio: float = 0.03,
        group_name: str = "sun_arrows",
    ) -> None:
        sd = np.asarray(sun_dirs, dtype=np.float64)
        if sd.ndim != 2 or sd.shape[1] != 3:
            raise ValueError("sun_dirs必须是(S,3)")

        s = _meters_to_su_internal_inch(units)
        L = float(length_m) * s
        if L <= 0.0:
            raise ValueError("length_m必须>0")

        o = np.array([float(origin[0]), float(origin[1]), float(origin[2])], dtype=np.float64)

        edge_idx: list[int] = []
        for i in range(int(sd.shape[0])):
            _attach_one_arrow(
                geo,
                origin=o,
                direction=sd[i],
                L=L,
                head_len_ratio=float(head_len_ratio),
                head_radius_ratio=float(head_radius_ratio),
                out_edge_idx=edge_idx,
            )

        # 记录到geo上，给exportSU用（不改Geo数据结构，动态挂属性）
        eg = getattr(geo, "edge_groups", None)
        if eg is None or (not isinstance(eg, dict)):
            eg = {}
            setattr(geo, "edge_groups", eg)
        eg[str(group_name)] = edge_idx