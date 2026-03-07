# exportSU.py
from __future__ import annotations

import os
from typing import Any, List, Tuple
import ctypes as ct

import numpy as np

import obj._SU_API as SU
from obj.Geo import SketchUpGeo
from util.geo_sampling import GeoSampling

from util.exportSU_util import *

# =========================
# 仅对外暴露：export_geo_tex(dll_path, geo, out_skp, geo4=None)
# =========================
__all__ = ["export_geo_tex"]


class _Exporter:
    """
    负责：
    - 用SketchUpGeo创建面（outer+inner loops，洞必须空）
    - 贴图用face bbox bound映射，与面边数无关
    - 导出SketchUpGeo.edges（线）
    - 可选：每个face单独放进一个Group，减少SUFacePositionMaterial互相干扰
    """

    def __init__(self, sketchup_api_dll: str) -> None:
        self.api = SketchUpAPI(sketchup_api_dll)

    def _loop_verts_to_xyz(self, geo: SketchUpGeo, loop_idx: int) -> List[Tuple[float, float, float]]:
        if loop_idx < 0 or loop_idx >= len(geo.loops):
            raise ValueError(f"loop索引越界：{loop_idx}, loops={len(geo.loops)}")
        lp = geo.loops[int(loop_idx)]
        vids = list(lp.verts)
        if len(vids) < 3:
            raise ValueError(f"loop.verts长度必须>=3，当前len={len(vids)}")

        verts: List[Tuple[float, float, float]] = []
        for vi in vids:
            if int(vi) < 0 or int(vi) >= len(geo.points):
                raise ValueError(f"point索引越界：{vi}, points={len(geo.points)}")
            p = geo.points[int(vi)]
            verts.append((float(p.x), float(p.y), float(p.z)))
        return verts

    def _create_face_outer(self, entities: SU.SUEntitiesRef, verts: List[Tuple[float, float, float]]) -> SU.SUFaceRef:
        d = self.api.dll
        n = int(len(verts))
        if n < 3:
            raise ValueError(f"face outer verts必须>=3个点，当前n={n}")

        loop = SU.invalid_ref(SU.SULoopInputRef)
        SU.su_check(d.SULoopInputCreate(ct.byref(loop)), "SULoopInputCreate")
        for i in range(n):
            SU.su_check(d.SULoopInputAddVertexIndex(loop, ct.c_size_t(i)), "SULoopInputAddVertexIndex")

        pts = (SU.SUPoint3D * n)()
        for k in range(n):
            x, y, z = verts[k]
            pts[k] = SU.SUPoint3D(float(x), float(y), float(z))

        face = SU.invalid_ref(SU.SUFaceRef)
        SU.su_check(d.SUFaceCreate(ct.byref(face), pts, ct.byref(loop)), "SUFaceCreate")
        SU.su_check(d.SUEntitiesAddFaces(entities, ct.c_size_t(1), ct.byref(face)), "SUEntitiesAddFaces")
        return face

    def _add_inner_loop(self, face: SU.SUFaceRef, verts: List[Tuple[float, float, float]]) -> None:
        d = self.api.dll
        if not hasattr(d, "SUFaceAddInnerLoop"):
            raise AttributeError("SketchUpAPI.dll缺少SUFaceAddInnerLoop，无法导出带洞面")

        n = int(len(verts))
        if n < 3:
            raise ValueError(f"inner loop verts必须>=3个点，当前n={n}")

        loop = SU.invalid_ref(SU.SULoopInputRef)
        SU.su_check(d.SULoopInputCreate(ct.byref(loop)), "SULoopInputCreate")
        for i in range(n):
            SU.su_check(d.SULoopInputAddVertexIndex(loop, ct.c_size_t(i)), "SULoopInputAddVertexIndex")

        pts = (SU.SUPoint3D * n)()
        for k in range(n):
            x, y, z = verts[k]
            pts[k] = SU.SUPoint3D(float(x), float(y), float(z))

        SU.su_check(d.SUFaceAddInnerLoop(face, pts, ct.byref(loop)), "SUFaceAddInnerLoop")

    def _create_face_with_holes(self, entities: SU.SUEntitiesRef, geo: SketchUpGeo, fi: int) -> SU.SUFaceRef:
        f = geo.faces[int(fi)]
        basis = GeoSampling.face_plane_basis(geo, int(fi))

        outer = self._loop_verts_to_xyz(geo, int(f.outer_loop))
        outer = ensure_ccw_in_basis(outer, basis.p0, basis.u, basis.v)
        face_ref = self._create_face_outer(entities, outer)

        inner_loops = list(getattr(f, "inner_loops", []))
        for li in inner_loops:
            hole = self._loop_verts_to_xyz(geo, int(li))
            hole = ensure_cw_in_basis(hole, basis.p0, basis.u, basis.v)
            self._add_inner_loop(face_ref, hole)

        return face_ref

    def _create_edge(
        self,
        entities: SU.SUEntitiesRef,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> None:
        d = self.api.dll
        if not hasattr(d, "SUEdgeCreate") or not hasattr(d, "SUEntitiesAddEdges"):
            raise AttributeError("SketchUpAPI.dll缺少SUEdgeCreate或SUEntitiesAddEdges，无法导出线段")

        e = SU.invalid_ref(SU.SUEdgeRef)
        pa = SU.SUPoint3D(float(a[0]), float(a[1]), float(a[2]))
        pb = SU.SUPoint3D(float(b[0]), float(b[1]), float(b[2]))
        SU.su_check(d.SUEdgeCreate(ct.byref(e), pa, pb), "SUEdgeCreate")
        SU.su_check(d.SUEntitiesAddEdges(entities, ct.c_size_t(1), ct.byref(e)), "SUEntitiesAddEdges")

    def _create_group_entities(
        self,
        parent_entities: SU.SUEntitiesRef,
        *,
        name: str,
    ) -> SU.SUEntitiesRef:
        """
        在parent_entities中创建一个Group，返回该Group的EntitiesRef。
        """
        d = self.api.dll
        if not hasattr(d, "SUGroupCreate") or not hasattr(d, "SUEntitiesAddGroup") or not hasattr(d, "SUGroupGetEntities"):
            raise AttributeError("SketchUpAPI.dll缺少SUGroupCreate/SUEntitiesAddGroup/SUGroupGetEntities，无法按face分组导出")

        grp = SU.invalid_ref(SU.SUGroupRef)
        SU.su_check(d.SUGroupCreate(ct.byref(grp)), "SUGroupCreate")

        if hasattr(d, "SUGroupSetName"):
            SU.su_check(d.SUGroupSetName(grp, str(name).encode("utf-8")), "SUGroupSetName")

        SU.su_check(d.SUEntitiesAddGroup(parent_entities, grp), "SUEntitiesAddGroup")

        ge = SU.invalid_ref(SU.SUEntitiesRef)
        SU.su_check(d.SUGroupGetEntities(grp, ct.byref(ge)), "SUGroupGetEntities")
        return ge

    def _position_material_on_face_by_bound(
        self,
        face: SU.SUFaceRef,
        *,
        fi: int,
        front: bool,
        material: SU.SUMaterialRef,
        basis_p0: np.ndarray,
        basis_u: np.ndarray,
        basis_v: np.ndarray,
        basis_n: np.ndarray,
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
    ) -> None:
        corners3 = bound_corners_3d(basis_p0, basis_u, basis_v, u_min, u_max, v_min, v_max)
        uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

        mapping = SU.SUMaterialPositionInput()
        mapping.num_uv_coords = ct.c_size_t(4)

        for i in range(4):
            x, y, z = corners3[i]
            mapping.points[i] = SU.SUPoint3D(float(x), float(y), float(z))
            uu, vv = uvs[i]
            mapping.uv_coords[i] = SU.SUPoint2D(float(uu), float(vv))

        mapping.material = material
        nn = normalize(basis_n.astype(np.float64, copy=False))
        mapping.projection = SU.SUVector3D(float(nn[0]), float(nn[1]), float(nn[2]))

        d = self.api.dll
        res = d.SUFacePositionMaterial(face, ct.c_bool(bool(front)), ct.byref(mapping))
        if int(res) != int(SU.SU_ERROR_NONE):
            side = "front" if bool(front) else "back"
            print(f"[W] SUFacePositionMaterial failed: fi={int(fi)} side={side} SUResult={int(res)}")
            return

    def export_geo_tex(
        self,
        geo: SketchUpGeo,
        out_skp: str,
        *,
        geo4: Any = None,
        group_per_face: bool = True,
    ) -> str:
        if not isinstance(geo, SketchUpGeo):
            raise TypeError("geo必须是SketchUpGeo")
        if len(geo.faces) == 0:
            raise ValueError("geo.faces为空")

        d = self.api.dll

        model = SU.invalid_ref(SU.SUModelRef)
        SU.su_check(d.SUModelCreate(ct.byref(model)), "SUModelCreate")

        root_entities = SU.invalid_ref(SU.SUEntitiesRef)
        SU.su_check(d.SUModelGetEntities(model, ct.byref(root_entities)), "SUModelGetEntities")

        used_mat: List[int] = []
        used_set = set()
        for f in geo.faces:
            for mi in (int(f.front_material), int(f.back_material)):
                if mi < 0:
                    continue
                if mi >= len(geo.materials):
                    raise ValueError(f"face引用material越界：{mi}, materials={len(geo.materials)}")
                if int(geo.materials[mi].texture_index) < 0:
                    continue
                if mi not in used_set:
                    used_set.add(mi)
                    used_mat.append(mi)

        if len(used_mat) == 0:
            raise ValueError("没有任何face使用带Texture的Material（material.texture_index>=0）")

        mat_model_sizes: dict[int, Tuple[float, float]] = {mi: (1.0, 1.0) for mi in used_mat}

        for fi, f in enumerate(geo.faces):
            fi = int(fi)
            basis = GeoSampling.face_plane_basis(geo, fi)

            outer = self._loop_verts_to_xyz(geo, int(f.outer_loop))
            outer = ensure_ccw_in_basis(outer, basis.p0, basis.u, basis.v)

            b_geo4, ok_geo4 = get_geo4_bounds_for_face(geo4, fi)
            if ok_geo4 and b_geo4 is not None:
                u_min, u_max, v_min, v_max = b_geo4
                ok = True
            else:
                u_min, u_max, v_min, v_max, ok = bounds_uv_from_verts(outer, basis.p0, basis.u, basis.v)

            if not ok:
                continue

            w = max(u_max - u_min, 1e-6)
            h = max(v_max - v_min, 1e-6)

            for mi in (int(f.front_material), int(f.back_material)):
                if mi in mat_model_sizes:
                    cur_w, cur_h = mat_model_sizes[mi]
                    if (w * h) > (cur_w * cur_h):
                        mat_model_sizes[mi] = (float(w), float(h))

        for mi in used_mat:
            ti = int(geo.materials[mi].texture_index)
            if ti < 0 or ti >= len(geo.textures):
                raise ValueError(f"material.texture_index越界：{ti}, textures={len(geo.textures)}")
            tp = str(geo.textures[ti].png)
            if not os.path.exists(tp):
                raise FileNotFoundError(tp)

        mat_ref_by_index: List[SU.SUMaterialRef] = [SU.invalid_ref(SU.SUMaterialRef) for _ in range(len(geo.materials))]
        created_mats: List[SU.SUMaterialRef] = []

        for mi in used_mat:
            m = geo.materials[mi]
            ti = int(m.texture_index)
            tex = geo.textures[ti]

            mat_ref = SU.invalid_ref(SU.SUMaterialRef)
            SU.su_check(d.SUMaterialCreate(ct.byref(mat_ref)), "SUMaterialCreate")

            name = m.name if getattr(m, "name", "") else f"mat_{mi:03d}"
            SU.su_check(d.SUMaterialSetName(mat_ref, str(name).encode("utf-8")), "SUMaterialSetName")

            tex_ref = SU.invalid_ref(SU.SUTextureRef)
            model_w, model_h = mat_model_sizes[mi]
            SU.su_check(
                d.SUTextureCreateFromFile(
                    ct.byref(tex_ref),
                    str(tex.png).encode("utf-8"),
                    ct.c_double(float(model_w)),
                    ct.c_double(float(model_h)),
                ),
                "SUTextureCreateFromFile",
            )
            SU.su_check(d.SUMaterialSetTexture(mat_ref, tex_ref), "SUMaterialSetTexture")

            mat_ref_by_index[mi] = mat_ref
            created_mats.append(mat_ref)

        mats_arr = (SU.SUMaterialRef * len(created_mats))()
        for i, mr in enumerate(created_mats):
            mats_arr[i] = mr
        SU.su_check(d.SUModelAddMaterials(model, ct.c_size_t(len(created_mats)), mats_arr), "SUModelAddMaterials")

        for fi, f in enumerate(geo.faces):
            fi = int(fi)

            if bool(group_per_face):
                grp_entities = self._create_group_entities(root_entities, name=f"face_{fi:06d}")
                entities_for_face = grp_entities
            else:
                entities_for_face = root_entities

            face_ref = self._create_face_with_holes(entities_for_face, geo, fi)

            fm = int(f.front_material)
            if fm >= 0:
                if fm >= len(mat_ref_by_index):
                    raise ValueError(f"face[{fi}] front_material越界：{fm}")
                mr = mat_ref_by_index[fm]
                SU.su_check(d.SUFaceSetFrontMaterial(face_ref, mr), "SUFaceSetFrontMaterial")

                basis = GeoSampling.face_plane_basis(geo, fi)

                b_geo4, ok_geo4 = get_geo4_bounds_for_face(geo4, fi)
                if ok_geo4 and b_geo4 is not None:
                    u_min, u_max, v_min, v_max = b_geo4
                    ok = True
                else:
                    outer = self._loop_verts_to_xyz(geo, int(f.outer_loop))
                    outer = ensure_ccw_in_basis(outer, basis.p0, basis.u, basis.v)
                    u_min, u_max, v_min, v_max, ok = bounds_uv_from_verts(outer, basis.p0, basis.u, basis.v)

                if ok:
                    self._position_material_on_face_by_bound(
                        face_ref,
                        fi=fi,
                        front=True,
                        material=mr,
                        basis_p0=basis.p0.astype(np.float64, copy=False),
                        basis_u=basis.u.astype(np.float64, copy=False),
                        basis_v=basis.v.astype(np.float64, copy=False),
                        basis_n=basis.n.astype(np.float64, copy=False),
                        u_min=float(u_min),
                        u_max=float(u_max),
                        v_min=float(v_min),
                        v_max=float(v_max),
                    )

        edge_groups = getattr(geo, "edge_groups", None)
        grouped = set()

        if isinstance(edge_groups, dict) and edge_groups:
            for gname, ei_list in edge_groups.items():
                grp_entities = self._create_group_entities(root_entities, name=str(gname))
                for ei in ei_list:
                    ei = int(ei)
                    if ei < 0 or ei >= len(geo.edges):
                        raise ValueError(f"edge_groups[{gname}]含越界edge索引：{ei}, edges={len(geo.edges)}")
                    e = geo.edges[ei]
                    a_i = int(e.a)
                    b_i = int(e.b)
                    if a_i < 0 or a_i >= len(geo.points):
                        raise ValueError(f"edge[{ei}].a越界：{a_i}, points={len(geo.points)}")
                    if b_i < 0 or b_i >= len(geo.points):
                        raise ValueError(f"edge[{ei}].b越界：{b_i}, points={len(geo.points)}")
                    pa = geo.points[a_i]
                    pb = geo.points[b_i]
                    self._create_edge(
                        grp_entities,
                        (float(pa.x), float(pa.y), float(pa.z)),
                        (float(pb.x), float(pb.y), float(pb.z)),
                    )
                    grouped.add(ei)

        for ei, e in enumerate(geo.edges):
            if ei in grouped:
                continue
            a_i = int(e.a)
            b_i = int(e.b)
            if a_i < 0 or a_i >= len(geo.points):
                raise ValueError(f"edge[{ei}].a越界：{a_i}, points={len(geo.points)}")
            if b_i < 0 or b_i >= len(geo.points):
                raise ValueError(f"edge[{ei}].b越界：{b_i}, points={len(geo.points)}")
            pa = geo.points[a_i]
            pb = geo.points[b_i]
            self._create_edge(
                root_entities,
                (float(pa.x), float(pa.y), float(pa.z)),
                (float(pb.x), float(pb.y), float(pb.z)),
            )

        out_parent = os.path.dirname(os.path.abspath(out_skp))
        if out_parent and not os.path.exists(out_parent):
            os.makedirs(out_parent, exist_ok=True)

        SU.su_check(d.SUModelSaveToFile(model, out_skp.encode("utf-8")), "SUModelSaveToFile")
        SU.su_check(d.SUModelRelease(ct.byref(model)), "SUModelRelease")
        SU.su_check(d.SUTerminate(), "SUTerminate")
        return out_skp


def export_geo_tex(
    sketchup_api_dll: str,
    geo: SketchUpGeo,
    out_skp: str,
    *,
    geo4: Any = None,
    group_per_face: bool = True,
) -> str:
    """
    只暴露这一个函数。

    - out_skp必须是输出skp文件路径
    - geo4可选：传Geo4RayTest时，会优先用geo4.faceBoundUVL/faceBoundValidL做贴图bound定位
    - group_per_face：每个face单独一个group，减少SUFacePositionMaterial互相干扰
    """
    if not str(out_skp).lower().endswith(".skp"):
        raise ValueError("out_skp必须以.skp结尾")

    exp = _Exporter(sketchup_api_dll)
    return exp.export_geo_tex(geo, str(out_skp), geo4=geo4, group_per_face=bool(group_per_face))