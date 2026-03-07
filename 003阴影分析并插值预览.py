# 003阴影分析并插值预览.py
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import taichi as ti

from readSU import read_su_mesh, step_m_to_model_units
from util.geo_sampling import GeoSampling
from plot.plot采样点预览 import SamplePreview
from plot.plot采样点强度插值预览 import SampleIntensityInterpPreview
from util.ray_manager import RayManager
from util.shadowTaichi import analyze_shadow_visibility
from obj.Geo4RayTest import Geo4RayTest
from util.img_bake import bake_faces_bilinear, attach_baked_textures_to_geo


def _build_face_normals(geo) -> np.ndarray:
    fn = np.zeros((len(geo.faces), 3), dtype=np.float32)
    for i, f in enumerate(geo.faces):
        fn[i, 0] = float(f.n.x)
        fn[i, 1] = float(f.n.y)
        fn[i, 2] = float(f.n.z)
    nrm = np.linalg.norm(fn, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12).astype(np.float32)
    fn = (fn / nrm).astype(np.float32)
    return fn


def main() -> None:
    dll_path = r"D:\Program Files\SketchUp\SketchUp2025\SketchUp\SketchUpAPI.dll"
    skp_path = r"C:\Users\L\Desktop\可园-用于日照分析.skp"

    step_m = 0.1
    timezone_str = "Asia/Shanghai"
    lat = 23.1291
    lon = 113.2644
    base = datetime(2026, 6, 21, 6, 0, 0)
    timetable = [base + timedelta(hours=i) for i in range(12)]

    sun_dirs = RayManager.build(
        "solstice_day",
        timezone_str=timezone_str,
        lat=lat,
        lon=lon,
        start_dt=base,
        hours=len(timetable),
        step_hours=1,
    )

    sun_dirs = RayManager.build(
        "year_hourly_2026",
        timezone_str=timezone_str,
        lat=lat,
        lon=lon,
    )

    sun_dirs = RayManager.build(
        "sky_view_factor",
        azi_deg_range=(0.0, 360.0),
        zenith_deg_range=(0.0, 90.0),
        target_count=200,
        seed=0,
    )

    geo = read_su_mesh(dll_path, skp_path, max_depth=64, verbose=False)
    units = geo.units
    step_model = step_m_to_model_units(step_m, units) / 0.0254

    # 一次打包：采样点+三角网+tri->face映射+faceBoundUV
    geo4 = Geo4RayTest.build(geo, step_model)

    face_normals = _build_face_normals(geo)
    ptVisL_Sample = analyze_shadow_visibility(
        pts=geo4.ptL_Sample.astype(np.float32, copy=False),
        face_idx=geo4.ptOnfaceIdxL_Sample.astype(np.int32, copy=False),
        is_on=geo4.isOnFaceL_Sample.astype(np.int32, copy=False),
        face_normals=face_normals.astype(np.float32, copy=False),
        sun_dirs=sun_dirs.astype(np.float32, copy=False),
        v0ontriL=geo4.v0ontriL.astype(np.float32, copy=False),
        v1ontriL=geo4.v1ontriL.astype(np.float32, copy=False),
        v2ontriL=geo4.v2ontriL.astype(np.float32, copy=False),
        triOnfaceIdxL=geo4.triOnfaceIdxL.astype(np.int32, copy=False),
        eps=1e-3,
        ti_arch=ti.cuda,
    )

    # 先在show_points外面做bake+attach
    tex_size = 256
    out_dir = "_temp_sunshade_interp"
    vis_clamp = (0.0, 1.0)
    cmap_name = "turbo"

    face_to_png, face_to_scalar = bake_faces_bilinear(
        geo,
        geo4.ptL_Sample,
        geo4.ptOnfaceIdxL_Sample,
        geo4.isOnFaceL_Sample,
        ptVisL_Sample,
        geo4.faceBoundUVL,
        geo4.faceBoundValidL,
        tex_size=tex_size,
        out_dir=out_dir,
        clamp=vis_clamp,
        cmap_name=cmap_name,
    )

    attach_baked_textures_to_geo(geo, face_to_png, geo4.faceBoundUVL)

    # 再做插值贴图三维预览
    SampleIntensityInterpPreview.show_points(
        geo,
        geo4,
        face_to_scalar,
        title="采样点强度插值预览ptVis",
        tex_size=tex_size,
        vis_clamp=vis_clamp,
        cmap_name=cmap_name,
    )

    print("sun_dirs:", int(sun_dirs.shape[0]))
    print("sample points:", int(geo4.ptL_Sample.shape[0]), "onFace:", int(np.sum(geo4.isOnFaceL_Sample)))
    print("triangles:", int(geo4.v0ontriL.shape[0]))
    print("ptVis min/max:", float(np.min(ptVisL_Sample)), float(np.max(ptVisL_Sample)))


if __name__ == "__main__":
    main()