# 002采样点生成.py
from __future__ import annotations

from datetime import datetime, timedelta

from readSU import read_su_mesh, step_m_to_model_units
from util.geo_sampling import GeoSampling
from plot.plot采样点预览 import SamplePreview


# 如果你项目里已有RayManager，这段保留；没有也不影响采样功能（不使用sun_dirs也不影响）。
# from ray_manager import RayManager


def main() -> None:
    dll_path = r"D:\Program Files\SketchUp\SketchUp2025\SketchUp\SketchUpAPI.dll"
    skp_path = r"C:\Users\L\Desktop\可园-用于日照分析.skp"

    step_m = 0.5
    timezone_str = "Asia/Shanghai"
    lat = 23.1291
    lon = 113.2644
    base = datetime(2026, 6, 21, 6, 0, 0)
    timetable = [base + timedelta(hours=i) for i in range(12)]

    # 2026年夏至日白天模式：
    # sun_dirs = RayManager.build(
    #     "solstice_day",
    #     timezone_str=timezone_str,
    #     lat=lat,
    #     lon=lon,
    # )

    geo = read_su_mesh(dll_path, skp_path, max_depth=64, verbose=False)
    units = geo.units
    step_model = step_m_to_model_units(step_m, units) / 0.0254
    ptL, faceIdxL, isOnFaceL = GeoSampling.sample_all_faces(geo, step_model)

    # 你要的三个输出就在这里：
    print("ptL:", ptL.shape, "faceIdxL:", faceIdxL.shape, "isOnFaceL:", isOnFaceL.shape)

    # 3D预览
    SamplePreview.show_points(
        ptL,
        faceIdxL,
        isOnFaceL,
        title="采样点预览(按faceIdx着色，非面内点alpha=0.4)",
    )


if __name__ == "__main__":
    main()