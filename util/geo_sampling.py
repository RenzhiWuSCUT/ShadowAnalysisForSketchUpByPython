# geo_sampling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from obj.Geo import SketchUpGeo, Point3


@dataclass(frozen=True)
class PlaneBasis:
    p0: np.ndarray  # (3,)
    n: np.ndarray   # (3,)
    u: np.ndarray   # (3,)
    v: np.ndarray   # (3,)


class GeoSampling:
    @staticmethod
    def step_m_to_model_units(step_m: float, units: str) -> float:
        u = (units or "").strip()
        if u == "Meters":
            return float(step_m)
        if u == "Centimeters":
            return float(step_m) * 100.0
        if u == "Millimeters":
            return float(step_m) * 1000.0
        if u == "Feet":
            return float(step_m) / 0.3048
        if u == "Inches":
            return float(step_m) / 0.0254
        # 未知单位：按米处理（直接暴露，不做隐藏兜底）
        return float(step_m)

    @staticmethod
    def _p3_to_np(p: Point3) -> np.ndarray:
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v
        return v / n

    @staticmethod
    def face_plane_basis(geo: SketchUpGeo, face_idx: int) -> PlaneBasis:
        face = geo.faces[int(face_idx)]
        outer_lp = geo.loops[face.outer_loop]
        if len(outer_lp.verts) < 3:
            raise ValueError(f"face {face_idx} outer loop verts < 3")

        p0 = GeoSampling._p3_to_np(geo.points[outer_lp.verts[0]])

        n = np.array([float(face.n.x), float(face.n.y), float(face.n.z)], dtype=np.float64)
        if float(np.linalg.norm(n)) == 0.0:
            # 退化面：用outer loop前三点算一次
            p1 = GeoSampling._p3_to_np(geo.points[outer_lp.verts[1]])
            p2 = GeoSampling._p3_to_np(geo.points[outer_lp.verts[2]])
            n = np.cross(p1 - p0, p2 - p0)
        n = GeoSampling._normalize(n)
        if float(np.linalg.norm(n)) == 0.0:
            raise ValueError(f"face {face_idx} normal is zero")

        # 选一个不平行于n的参考轴，构造u,v
        ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ax, n))) > 0.95:
            ax = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        u = np.cross(ax, n)
        u = GeoSampling._normalize(u)
        if float(np.linalg.norm(u)) == 0.0:
            raise ValueError(f"face {face_idx} cannot build u axis")
        v = np.cross(n, u)
        v = GeoSampling._normalize(v)

        return PlaneBasis(p0=p0, n=n, u=u, v=v)

    @staticmethod
    def _loop_points_3d(geo: SketchUpGeo, loop_idx: int) -> np.ndarray:
        lp = geo.loops[int(loop_idx)]
        if len(lp.verts) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        pts = [GeoSampling._p3_to_np(geo.points[vi]) for vi in lp.verts]
        return np.stack(pts, axis=0).astype(np.float64)

    @staticmethod
    def _project_to_2d(basis: PlaneBasis, pts3: np.ndarray) -> np.ndarray:
        # 坐标定义：p = p0 + u*x + v*y + n*z；这里取(x,y)
        d = pts3 - basis.p0[None, :]
        x = d @ basis.u
        y = d @ basis.v
        return np.stack([x, y], axis=1).astype(np.float64)

    @staticmethod
    def _point_in_ring_evenodd(pt: np.ndarray, poly: np.ndarray) -> bool:
        # poly: (M,2) 不要求闭合
        m = int(poly.shape[0])
        if m < 3:
            return False
        x = float(pt[0])
        y = float(pt[1])
        inside = False
        x0 = float(poly[-1, 0])
        y0 = float(poly[-1, 1])
        for i in range(m):
            x1 = float(poly[i, 0])
            y1 = float(poly[i, 1])

            # 边是否跨过水平射线
            cond = (y0 > y) != (y1 > y)
            if cond:
                # 交点x坐标
                t = (y - y0) / (y1 - y0)
                x_int = x0 + t * (x1 - x0)
                if x_int > x:
                    inside = not inside
            x0, y0 = x1, y1
        return inside

    @staticmethod
    def _point_on_face_2d(
        pt2: np.ndarray,
        outer2: np.ndarray,
        holes2: List[np.ndarray],
    ) -> bool:
        if not GeoSampling._point_in_ring_evenodd(pt2, outer2):
            return False
        for h in holes2:
            if GeoSampling._point_in_ring_evenodd(pt2, h):
                return False
        return True

    @staticmethod
    def sample_face_net(
        geo: SketchUpGeo,
        face_idx: int,
        step_model: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对单个face：
        1) 建平面基
        2) 取outer/inner loops投影到2D
        3) 用outer bbox生成渔网网格
        4) 判断每个网格点是否在face（outer内且不在holes内）
        输出：(pts3, faceIdx, isOnFace)
        pts3: (N,3) float64
        faceIdx: (N,) int32
        isOnFace: (N,) int32 (0/1)
        """
        basis = GeoSampling.face_plane_basis(geo, face_idx)
        face = geo.faces[int(face_idx)]
        outer3 = GeoSampling._loop_points_3d(geo, face.outer_loop)
        holes3 = [GeoSampling._loop_points_3d(geo, li) for li in face.inner_loops]
        outer2 = GeoSampling._project_to_2d(basis, outer3)
        holes2 = [GeoSampling._project_to_2d(basis, h3) for h3 in holes3 if h3.shape[0] >= 3]

        if outer2.shape[0] < 3:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        xs = outer2[:, 0]
        ys = outer2[:, 1]
        x_min = float(np.min(xs))
        x_max = float(np.max(xs))
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))

        step = float(step_model)
        if step <= 0.0:
            raise ValueError("step_model must be > 0")

        # 左上角 minPt: (x_min, y_max)
        # x向、y向按step扩展到首次超过bbox就停
        # 这里用包含端点的方式：最后一个点可能略超过，故用<= x_max + 1e-12
        nx = int(np.floor((x_max - x_min) / step)) + 1
        ny = int(np.floor((y_max - y_min) / step)) + 1

        if nx <= 0 or ny <= 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        x_grid = x_min + step * np.arange(nx, dtype=np.float64)
        y_grid = y_max - step * np.arange(ny, dtype=np.float64)  # 从上往下

        XX, YY = np.meshgrid(x_grid, y_grid, indexing="xy")
        pts2 = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1).astype(np.float64)

        is_on = np.zeros((pts2.shape[0],), dtype=np.int32)
        for i in range(pts2.shape[0]):
            if GeoSampling._point_on_face_2d(pts2[i], outer2, holes2):
                is_on[i] = 1

        # 还原到3D：p = p0 + u*x + v*y
        pts3 = (
            basis.p0[None, :]
            + pts2[:, 0:1] * basis.u[None, :]
            + pts2[:, 1:2] * basis.v[None, :]
        ).astype(np.float64)

        face_ids = np.full((pts3.shape[0],), int(face_idx), dtype=np.int32)
        return pts3, face_ids, is_on

    @staticmethod
    def sample_all_faces(
        geo: SketchUpGeo,
        step_model: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ptL: List[np.ndarray] = []
        faceIdxL: List[np.ndarray] = []
        isOnFaceL: List[np.ndarray] = []

        for fi in range(len(geo.faces)):
            p3, fids, on = GeoSampling.sample_face_net(geo, fi, step_model)
            if p3.shape[0] == 0:
                continue
            ptL.append(p3)
            faceIdxL.append(fids)
            isOnFaceL.append(on)

        if len(ptL) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        pts = np.concatenate(ptL, axis=0)
        fids = np.concatenate(faceIdxL, axis=0)
        on = np.concatenate(isOnFaceL, axis=0)
        return pts, fids, on