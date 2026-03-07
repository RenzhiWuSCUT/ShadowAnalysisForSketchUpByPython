# obj/SetGeo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .Geo import SketchUpGeo, Point3, Loop, Face, Material, Texture, UV, Edge


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v * 0.0
    return v / n


def _face_normal_from_verts(pts: Sequence[Tuple[float, float, float]]) -> Point3:
    if len(pts) < 3:
        return Point3(0.0, 0.0, 1.0)
    p0 = np.array(pts[0], dtype=np.float64)
    p1 = np.array(pts[1], dtype=np.float64)
    p2 = np.array(pts[2], dtype=np.float64)
    n = np.cross(p1 - p0, p2 - p0)
    n = _normalize(n)
    if float(np.linalg.norm(n)) <= 1e-12:
        return Point3(0.0, 0.0, 1.0)
    return Point3(float(n[0]), float(n[1]), float(n[2]))


@dataclass
class GeoBuilder:
    """
    只做一件事：构建SketchUpGeo（points/loops/faces/materials/textures/edges）。
    不依赖exportSU，不写SketchUp API，不做任何导出侧“兜底/降级”。
    """
    geo: SketchUpGeo

    @staticmethod
    def create(file_path: str = "(generated)", units: str = "Meters") -> "GeoBuilder":
        g = SketchUpGeo(file_path=file_path)
        g.units = str(units)
        return GeoBuilder(g)

    # -------------------------
    # 基础几何
    # -------------------------
    def add_point(self, x: float, y: float, z: float) -> int:
        self.geo.points.append(Point3(float(x), float(y), float(z)))
        return len(self.geo.points) - 1

    def add_points(self, pts: Iterable[Tuple[float, float, float]]) -> List[int]:
        out: List[int] = []
        for x, y, z in pts:
            out.append(self.add_point(float(x), float(y), float(z)))
        return out

    def add_edge(self, a: int, b: int) -> int:
        if a < 0 or a >= len(self.geo.points):
            raise ValueError(f"edge.a越界：{a}, points={len(self.geo.points)}")
        if b < 0 or b >= len(self.geo.points):
            raise ValueError(f"edge.b越界：{b}, points={len(self.geo.points)}")
        self.geo.edges.append(Edge(a=int(a), b=int(b)))
        return len(self.geo.edges) - 1

    # -------------------------
    # 材质/贴图
    # -------------------------
    def add_texture(self, png_path: str, uv: Sequence[Tuple[float, float]] | None = None) -> int:
        t = Texture(png=str(png_path), uv=[])
        if uv is not None:
            for u, v in uv:
                t.uv.append(UV(float(u), float(v)))
        self.geo.textures.append(t)
        return len(self.geo.textures) - 1

    def add_material_for_texture(self, texture_index: int, name: str | None = None) -> int:
        if texture_index < 0 or texture_index >= len(self.geo.textures):
            raise ValueError(f"texture_index越界：{texture_index}, textures={len(self.geo.textures)}")
        mname = str(name) if name else f"mat_{len(self.geo.materials):03d}"
        self.geo.materials.append(
            Material(
                name=mname,
                ptr=0,
                color_index=-1,
                texture_index=int(texture_index),
            )
        )
        return len(self.geo.materials) - 1

    # -------------------------
    # Loop/Face
    # -------------------------
    def add_outer_loop(self, vids: Sequence[int]) -> int:
        vids2 = [int(v) for v in vids]
        if len(vids2) < 3:
            raise ValueError(f"loop顶点数必须>=3，当前len={len(vids2)}")
        for v in vids2:
            if v < 0 or v >= len(self.geo.points):
                raise ValueError(f"loop顶点索引越界：{v}, points={len(self.geo.points)}")
        li = len(self.geo.loops)
        self.geo.loops.append(Loop(is_outer=True, edges=[], verts=vids2))
        return li

    def add_face(
        self,
        outer_loop: int,
        *,
        front_material: int,
        back_material: int = -1,
        inner_loops: Sequence[int] | None = None,
        set_face_normal: bool = True,
    ) -> int:
        if outer_loop < 0 or outer_loop >= len(self.geo.loops):
            raise ValueError(f"outer_loop越界：{outer_loop}, loops={len(self.geo.loops)}")
        if front_material >= len(self.geo.materials):
            raise ValueError(f"front_material越界：{front_material}, materials={len(self.geo.materials)}")
        if back_material >= len(self.geo.materials):
            raise ValueError(f"back_material越界：{back_material}, materials={len(self.geo.materials)}")

        il = list(inner_loops) if inner_loops is not None else []
        for li in il:
            if li < 0 or li >= len(self.geo.loops):
                raise ValueError(f"inner_loop越界：{li}, loops={len(self.geo.loops)}")

        f = Face(
            outer_loop=int(outer_loop),
            inner_loops=[int(x) for x in il],
            front_material=int(front_material),
            back_material=int(back_material),
        )

        if set_face_normal:
            vids = self.geo.loops[int(outer_loop)].verts
            pts = [(self.geo.points[v].x, self.geo.points[v].y, self.geo.points[v].z) for v in vids]
            f.n = _face_normal_from_verts(pts)

        self.geo.faces.append(f)
        return len(self.geo.faces) - 1

    def add_face_from_vids(
        self,
        vids: Sequence[int],
        *,
        front_material: int,
        back_material: int = -1,
        set_face_normal: bool = True,
    ) -> int:
        li = self.add_outer_loop(vids)
        return self.add_face(li, front_material=front_material, back_material=back_material, set_face_normal=set_face_normal)

    # -------------------------
    # 常用：线段箭头（“三菱柱箭头”）
    # -------------------------
    def add_arrow_edges(
        self,
        *,
        start: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        length: float,
        head_len_ratio: float = 0.18,
        head_radius_ratio: float = 0.07,
    ) -> None:
        dvec = np.array([float(direction[0]), float(direction[1]), float(direction[2])], dtype=np.float64)
        dn = float(np.linalg.norm(dvec))
        if dn <= 1e-12:
            raise ValueError("direction不能为零向量")
        ddir = dvec / dn

        L = float(length)
        if L <= 0.0:
            raise ValueError("length必须>0")

        head_len = max(L * float(head_len_ratio), 1e-6)
        head_len = min(head_len, L * 0.45)
        head_r = max(L * float(head_radius_ratio), 1e-6)

        p0 = np.array([float(start[0]), float(start[1]), float(start[2])], dtype=np.float64)
        tip = p0 + ddir * L
        base_center = tip - ddir * head_len

        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if float(np.linalg.norm(np.cross(ddir, tmp))) <= 1e-6:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = _normalize(np.cross(ddir, tmp))
        v = _normalize(np.cross(ddir, u))

        b0 = base_center + head_r * (u + v)
        b1 = base_center + head_r * (u - v)
        b2 = base_center + head_r * (-u - v)
        b3 = base_center + head_r * (-u + v)

        i_start = self.add_point(float(p0[0]), float(p0[1]), float(p0[2]))
        i_basec = self.add_point(float(base_center[0]), float(base_center[1]), float(base_center[2]))
        i_tip = self.add_point(float(tip[0]), float(tip[1]), float(tip[2]))

        i_b0 = self.add_point(float(b0[0]), float(b0[1]), float(b0[2]))
        i_b1 = self.add_point(float(b1[0]), float(b1[1]), float(b1[2]))
        i_b2 = self.add_point(float(b2[0]), float(b2[1]), float(b2[2]))
        i_b3 = self.add_point(float(b3[0]), float(b3[1]), float(b3[2]))

        self.add_edge(i_start, i_basec)
        self.add_edge(i_tip, i_b0)
        self.add_edge(i_tip, i_b1)
        self.add_edge(i_tip, i_b2)
        self.add_edge(i_tip, i_b3)


