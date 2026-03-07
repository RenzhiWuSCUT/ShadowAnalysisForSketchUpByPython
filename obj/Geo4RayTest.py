# Geo4RayTest.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from obj.Geo import SketchUpGeo, Point3
from util.geo_sampling import GeoSampling
from util.triangulateT import triangulate_all_faces


@dataclass(frozen=True)
class Geo4RayTest:
    # 采样点
    ptL_Sample: np.ndarray            # (P,3) float32/float64
    ptOnfaceIdxL_Sample: np.ndarray   # (P,) int32
    isOnFaceL_Sample: np.ndarray      # (P,) int32 0/1

    # 三角网（来自全模型三角剖分）
    v0ontriL: np.ndarray                # (T,3) float32
    v1ontriL: np.ndarray                # (T,3) float32
    v2ontriL: np.ndarray                # (T,3) float32
    triOnfaceIdxL: np.ndarray         # (T,) int32  每个三角面所属face

    # face在自身基面(u,v)上的bound（bbox）
    # faceBoundUVL[fi] = [u_min,u_max,v_min,v_max]
    faceBoundUVL: np.ndarray          # (F,4) float32
    faceBoundValidL: np.ndarray       # (F,) int32 0/1

    @staticmethod
    def _p3_to_np(p: Point3) -> np.ndarray:
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    @staticmethod
    def _loop_points_3d(geo: SketchUpGeo, loop_idx: int) -> np.ndarray:
        lp = geo.loops[int(loop_idx)]
        if len(lp.verts) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        pts = [Geo4RayTest._p3_to_np(geo.points[int(vi)]) for vi in lp.verts]
        return np.stack(pts, axis=0).astype(np.float64)

    @staticmethod
    def _project_to_uv(p0: np.ndarray, u: np.ndarray, v: np.ndarray, pts3: np.ndarray) -> np.ndarray:
        # p = p0 + u*U + v*V + n*W，这里取(U,V)
        d = pts3 - p0[None, :]
        uu = d @ u
        vv = d @ v
        return np.stack([uu, vv], axis=1).astype(np.float64)

    @staticmethod
    def _build_face_bounds_uv(geo: SketchUpGeo) -> tuple[np.ndarray, np.ndarray]:
        F = len(geo.faces)
        b = np.zeros((F, 4), dtype=np.float32)
        valid = np.zeros((F,), dtype=np.int32)

        for fi in range(F):
            face = geo.faces[int(fi)]
            outer3 = Geo4RayTest._loop_points_3d(geo, face.outer_loop)
            if outer3.shape[0] < 3:
                valid[fi] = 0
                continue

            basis = GeoSampling.face_plane_basis(geo, fi)
            outer2 = Geo4RayTest._project_to_uv(basis.p0, basis.u, basis.v, outer3)
            if outer2.shape[0] < 3:
                valid[fi] = 0
                continue

            u_min = float(np.min(outer2[:, 0]))
            u_max = float(np.max(outer2[:, 0]))
            v_min = float(np.min(outer2[:, 1]))
            v_max = float(np.max(outer2[:, 1]))

            if (u_max - u_min) <= 0.0 or (v_max - v_min) <= 0.0:
                # 极端退化也直接标无效，避免后续用bbox做快速剔除时出bug
                valid[fi] = 0
                continue

            b[fi, 0] = np.float32(u_min)
            b[fi, 1] = np.float32(u_max)
            b[fi, 2] = np.float32(v_min)
            b[fi, 3] = np.float32(v_max)
            valid[fi] = 1

        return b, valid

    @staticmethod
    def build(geo: SketchUpGeo, step_model: float) -> "Geo4RayTest":
        ptL, faceIdxL, isOnFaceL = GeoSampling.sample_all_faces(geo, float(step_model))
        tri_v0, tri_v1, tri_v2, tri_face = triangulate_all_faces(geo)

        faceBound_uv, faceBound_valid = Geo4RayTest._build_face_bounds_uv(geo)

        return Geo4RayTest(
            ptL_Sample=ptL.astype(np.float32, copy=False),  # (P,3)
            ptOnfaceIdxL_Sample=faceIdxL.astype(np.int32, copy=False),  # (P,)
            isOnFaceL_Sample=isOnFaceL.astype(np.int32, copy=False),  # (P,)
            v0ontriL=tri_v0.astype(np.float32, copy=False),  # (T,3)
            v1ontriL=tri_v1.astype(np.float32, copy=False),  # (T,3)
            v2ontriL=tri_v2.astype(np.float32, copy=False),  # (T,3)
            triOnfaceIdxL=tri_face.astype(np.int32, copy=False),  # (T,)
            faceBoundUVL=faceBound_uv.astype(np.float32, copy=False),  # (F,4)
            faceBoundValidL=faceBound_valid.astype(np.int32, copy=False),  # (F,)
        )