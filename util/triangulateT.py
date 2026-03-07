from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from util.geo_sampling import GeoSampling
from obj.Geo import SketchUpGeo

from shapely.geometry import Polygon, LinearRing, Point
from shapely.ops import triangulate


@dataclass
class _Face2D:
    basis_p0: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    outer2: np.ndarray
    holes2: List[np.ndarray]


def _signed_area(poly2: np.ndarray) -> float:
    x = poly2[:, 0].astype(np.float64, copy=False)
    y = poly2[:, 1].astype(np.float64, copy=False)
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - x2 * y))


def _ensure_ccw(poly2: np.ndarray) -> np.ndarray:
    if poly2.shape[0] < 3:
        return poly2
    if _signed_area(poly2) < 0.0:
        return poly2[::-1].copy()
    return poly2


def _ensure_cw(poly2: np.ndarray) -> np.ndarray:
    if poly2.shape[0] < 3:
        return poly2
    if _signed_area(poly2) > 0.0:
        return poly2[::-1].copy()
    return poly2


def _cleanup_ring(poly2: np.ndarray) -> np.ndarray:
    if poly2.shape[0] == 0:
        return poly2
    out = [poly2[0]]
    for i in range(1, poly2.shape[0]):
        if not np.allclose(poly2[i], out[-1]):
            out.append(poly2[i])
    poly2 = np.asarray(out, dtype=np.float64)
    if poly2.shape[0] >= 2 and np.allclose(poly2[0], poly2[-1]):
        poly2 = poly2[:-1]
    return poly2


def _loop_points_3d(geo: SketchUpGeo, loop_idx: int) -> np.ndarray:
    lp = geo.loops[int(loop_idx)]
    if len(lp.verts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = []
    for vi in lp.verts:
        p = geo.points[int(vi)]
        pts.append([float(p.x), float(p.y), float(p.z)])
    return np.asarray(pts, dtype=np.float64)


def _project_face_to_2d(geo: SketchUpGeo, face_idx: int) -> _Face2D:
    basis = GeoSampling.face_plane_basis(geo, face_idx)
    face = geo.faces[int(face_idx)]

    outer3 = _loop_points_3d(geo, face.outer_loop)
    holes3 = [_loop_points_3d(geo, li) for li in face.inner_loops]

    outer2 = GeoSampling._project_to_2d(basis, outer3)
    holes2 = [GeoSampling._project_to_2d(basis, h3) for h3 in holes3 if h3.shape[0] >= 3]

    outer2 = _cleanup_ring(outer2)
    holes2 = [_cleanup_ring(h2) for h2 in holes2]

    outer2 = _ensure_ccw(outer2)
    holes2 = [_ensure_cw(h2) for h2 in holes2]

    return _Face2D(
        basis_p0=basis.p0.astype(np.float64),
        basis_u=basis.u.astype(np.float64),
        basis_v=basis.v.astype(np.float64),
        outer2=outer2.astype(np.float64),
        holes2=[h.astype(np.float64) for h in holes2 if h.shape[0] >= 3],
    )


def _poly_to_shapely(outer2: np.ndarray, holes2: List[np.ndarray]) -> Polygon:
    outer = [(float(x), float(y)) for x, y in outer2]
    holes = [[(float(x), float(y)) for x, y in h] for h in holes2 if h.shape[0] >= 3]

    ring = LinearRing(outer)
    if not ring.is_ccw:
        outer = outer[::-1]

    fixed_holes = []
    for h in holes:
        r = LinearRing(h)
        if r.is_ccw:
            fixed_holes.append(h[::-1])
        else:
            fixed_holes.append(h)

    poly = Polygon(outer, holes=fixed_holes)

    # 常见“自交/重复点”修复：buffer(0)会暴露问题而不是悄悄吞掉
    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


def _triangulate_shapely(poly: Polygon) -> np.ndarray:
    """
    return (T,3,2) 三角形的2D坐标
    """
    if poly.is_empty:
        return np.zeros((0, 3, 2), dtype=np.float64)

    tris = triangulate(poly)

    out: List[np.ndarray] = []
    for t in tris:
        # triangulate有时会给出覆盖外包区域的片，需要筛掉洞外
        rp = t.representative_point()
        if not poly.covers(rp):
            continue
        coords = list(t.exterior.coords)
        if len(coords) < 4:
            continue
        p0 = coords[0]
        p1 = coords[1]
        p2 = coords[2]
        out.append(np.asarray([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.float64))

    if len(out) == 0:
        return np.zeros((0, 3, 2), dtype=np.float64)
    return np.stack(out, axis=0)


def triangulate_all_faces(geo: SketchUpGeo) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    输出：
    v0ontriL,v1ontriL,v2ontriL:(T,3) float32
    triOnfaceIdxL:(T,) int32
    """
    v0L: List[np.ndarray] = []
    v1L: List[np.ndarray] = []
    v2L: List[np.ndarray] = []
    fL: List[np.ndarray] = []

    for fi in range(len(geo.faces)):
        f2d = _project_face_to_2d(geo, fi)
        if f2d.outer2.shape[0] < 3:
            continue

        poly = _poly_to_shapely(f2d.outer2, f2d.holes2)
        tri2 = _triangulate_shapely(poly)  # (T,3,2)
        if tri2.shape[0] == 0:
            continue

        p0 = f2d.basis_p0
        u = f2d.basis_u
        v = f2d.basis_v

        a2 = tri2[:, 0, :]
        b2 = tri2[:, 1, :]
        c2 = tri2[:, 2, :]

        a3 = (p0[None, :] + a2[:, 0:1] * u[None, :] + a2[:, 1:2] * v[None, :]).astype(np.float32)
        b3 = (p0[None, :] + b2[:, 0:1] * u[None, :] + b2[:, 1:2] * v[None, :]).astype(np.float32)
        c3 = (p0[None, :] + c2[:, 0:1] * u[None, :] + c2[:, 1:2] * v[None, :]).astype(np.float32)

        v0L.append(a3)
        v1L.append(b3)
        v2L.append(c3)
        fL.append(np.full((tri2.shape[0],), int(fi), dtype=np.int32))

    if len(v0L) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    tri_v0 = np.concatenate(v0L, axis=0).astype(np.float32, copy=False)
    tri_v1 = np.concatenate(v1L, axis=0).astype(np.float32, copy=False)
    tri_v2 = np.concatenate(v2L, axis=0).astype(np.float32, copy=False)
    tri_face = np.concatenate(fL, axis=0).astype(np.int32, copy=False)
    return tri_v0, tri_v1, tri_v2, tri_face