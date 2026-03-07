# exportJson.py
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from obj.Geo import SketchUpGeo
from obj.Geo4RayTest import Geo4RayTest
from util.geo_sampling import GeoSampling


__all__ = ["export_face_stats_json"]


def _loop_verts_to_xyz(geo: SketchUpGeo, loop_idx: int) -> np.ndarray:
    if loop_idx < 0 or loop_idx >= len(geo.loops):
        raise ValueError(f"loop索引越界：{loop_idx}, loops={len(geo.loops)}")

    lp = geo.loops[int(loop_idx)]
    vids = list(lp.verts)
    if len(vids) < 3:
        return np.zeros((0, 3), dtype=np.float64)

    pts: List[List[float]] = []
    for vi in vids:
        if int(vi) < 0 or int(vi) >= len(geo.points):
            raise ValueError(f"point索引越界：{vi}, points={len(geo.points)}")
        p = geo.points[int(vi)]
        pts.append([float(p.x), float(p.y), float(p.z)])

    return np.asarray(pts, dtype=np.float64)


def _project_to_uv(
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pts3: np.ndarray,
) -> np.ndarray:
    if pts3.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    d = pts3 - p0[None, :]
    uu = d @ u
    vv = d @ v
    return np.stack([uu, vv], axis=1).astype(np.float64)


def _signed_area_2d(poly2: np.ndarray) -> float:
    n = int(poly2.shape[0])
    if n < 3:
        return 0.0

    x = poly2[:, 0]
    y = poly2[:, 1]
    s = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return 0.5 * s


def _polygon_area_abs(poly2: np.ndarray) -> float:
    return abs(_signed_area_2d(poly2))


def _face_area_in_su_internal_inch2(geo: SketchUpGeo, fi: int) -> float:
    if fi < 0 or fi >= len(geo.faces):
        raise ValueError(f"face索引越界：{fi}, faces={len(geo.faces)}")

    face = geo.faces[int(fi)]
    basis = GeoSampling.face_plane_basis(geo, int(fi))

    outer3 = _loop_verts_to_xyz(geo, face.outer_loop)
    if outer3.shape[0] < 3:
        return 0.0

    outer2 = _project_to_uv(basis.p0, basis.u, basis.v, outer3)
    area = _polygon_area_abs(outer2)

    for loop_idx in face.inner_loops:
        inner3 = _loop_verts_to_xyz(geo, int(loop_idx))
        if inner3.shape[0] < 3:
            continue
        inner2 = _project_to_uv(basis.p0, basis.u, basis.v, inner3)
        area -= _polygon_area_abs(inner2)

    if area < 0.0:
        area = 0.0
    return float(area)


def _area_scale_from_su_internal_inch2(units: str) -> float:
    u = (units or "").strip()

    if u == "Meters":
        return 0.0254 ** 2
    if u == "Centimeters":
        return 2.54 ** 2
    if u == "Millimeters":
        return 25.4 ** 2
    if u == "Feet":
        return (1.0 / 12.0) ** 2
    if u == "Inches":
        return 1.0

    raise ValueError(f"不支持的geo.units：{units}")


def _area_unit_text(units: str) -> str:
    u = (units or "").strip()

    if u == "Meters":
        return "m²"
    if u == "Centimeters":
        return "cm²"
    if u == "Millimeters":
        return "mm²"
    if u == "Feet":
        return "ft²"
    if u == "Inches":
        return "in²"

    raise ValueError(f"不支持的geo.units：{units}")


def _convert_face_area_to_geo_units(geo: SketchUpGeo, fi: int) -> float:
    area_in2 = _face_area_in_su_internal_inch2(geo, fi)
    scale = _area_scale_from_su_internal_inch2(geo.units)
    return float(area_in2 * scale)


def _round_to_10_percent_bucket(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    pct = x * 100.0
    bucket = np.round(pct / 10.0) * 10.0
    bucket = np.clip(bucket, 0.0, 100.0).astype(np.int32)
    return bucket


def _build_bucket_dict(bucket_pct: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {f"{i}%": 0 for i in range(0, 101, 10)}
    if bucket_pct.size == 0:
        return out

    for k in range(0, 101, 10):
        out[f"{k}%"] = int(np.sum(bucket_pct == k))
    return out


def _collect_face_stats(
    geo: SketchUpGeo,
    geo4: Geo4RayTest,
    pt_vis: np.ndarray,
) -> List[Dict]:
    pt_vis = np.asarray(pt_vis, dtype=np.float64).reshape(-1)

    if pt_vis.shape[0] != geo4.ptL_Sample.shape[0]:
        raise ValueError(
            f"pt_vis长度必须等于采样点数，当前pt_vis={pt_vis.shape[0]}, "
            f"sample_points={geo4.ptL_Sample.shape[0]}"
        )

    face_idx_all = geo4.ptOnfaceIdxL_Sample.astype(np.int32, copy=False)
    is_on_all = geo4.isOnFaceL_Sample.astype(np.int32, copy=False)

    area_key = f"face_area({_area_unit_text(geo.units)})"

    result: List[Dict] = []

    for fi in range(len(geo.faces)):
        mask = (face_idx_all == int(fi)) & (is_on_all == 1)
        vals = pt_vis[mask]
        vals = vals[np.isfinite(vals)]

        buckets = _round_to_10_percent_bucket(vals)
        bucket_counts = _build_bucket_dict(buckets)

        item = {
            "face_index": int(fi),
            area_key: float(_convert_face_area_to_geo_units(geo, fi)),
            "valid_point_count": int(vals.shape[0]),
            "percent_bucket_counts": bucket_counts,
        }
        result.append(item)

    return result


def export_face_stats_json(
    geo: SketchUpGeo,
    geo4: Geo4RayTest,
    pt_vis: np.ndarray,
    out_skp: str,
) -> str:
    out_skp_abs = os.path.abspath(out_skp)
    out_dir = os.path.dirname(out_skp_abs)
    skp_stem = os.path.splitext(os.path.basename(out_skp_abs))[0]
    out_json = os.path.join(out_dir, f"{skp_stem}.json")

    data = {
        "source_skp": os.path.basename(out_skp_abs),
        "geo_units": str(geo.units),
        "face_count": int(len(geo.faces)),
        "sample_point_count": int(geo4.ptL_Sample.shape[0]),
        "items": _collect_face_stats(geo, geo4, pt_vis),
    }


    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_json