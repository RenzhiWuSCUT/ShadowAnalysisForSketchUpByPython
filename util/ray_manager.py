# ray_manager.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from util.sun_utils import build_sun_dirs


@dataclass(frozen=True)
class RayBuildResult:
    sun_dirs: np.ndarray
    timetable: Optional[List[datetime]] = None  # 时间类模式才有


class RayManager:
    """
    射线管理器:统一生成sun_dirs(N,3)

    mode取值:
    1) solstice_day
       2026年夏至日白天逐时(默认:2026-06-21 06:00开始,12小时,步长1小时)
       需要参数:timezone_str,lat,lon
       可选参数:start_dt,hours,step_hours

    2) year_hourly_2026
       2026年逐时(8760小时)
       需要参数:timezone_str,lat,lon
       可选参数:start_dt,end_dt

    3) sky_view_factor
       天空角系数射线:按给定azi/zenith范围与步长生成半球射线,再均匀抽样到约200条
       需要参数:azi_deg_range,zenith_deg_range
       可选参数:azi_step_deg,zenith_step_deg,target_count,seed
       注意:azi按从北向顺时针,zenith从天顶向下(0在正上,90在地平)
    """

    @staticmethod
    def build(
        mode: str,
        *,
        timezone_str: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        # solstice_day
        start_dt: Optional[datetime] = None,
        hours: int = 12,
        step_hours: int = 1,
        # year_hourly_2026
        end_dt: Optional[datetime] = None,
        # sky_view_factor
        azi_deg_range: Tuple[float, float] = (0.0, 360.0),
        zenith_deg_range: Tuple[float, float] = (0.0, 90.0),
        azi_step_deg: Optional[float] = None,
        zenith_step_deg: Optional[float] = None,
        target_count: int = 200,
        seed: int = 0,
    ) -> np.ndarray:
        """
        统一入口:输入模式名,输出sun_dirs(float32)

        - solstice_day/year_hourly_2026会调用build_sun_dirs,并自动过滤地平线下射线
        - sky_view_factor为半球采样,不需要timezone/lat/lon
        """
        if mode == "solstice_day":
            res = RayManager._solstice_day(
                timezone_str=timezone_str,
                lat=lat,
                lon=lon,
                start_dt=start_dt,
                hours=hours,
                step_hours=step_hours,
            )
            return res.sun_dirs

        if mode == "year_hourly_2026":
            res = RayManager._year_hourly_2026(
                timezone_str=timezone_str,
                lat=lat,
                lon=lon,
                start_dt=start_dt,
                end_dt=end_dt,
            )
            return res.sun_dirs

        if mode == "sky_view_factor":
            return RayManager._sky_view_factor(
                azi_deg_range=azi_deg_range,
                zenith_deg_range=zenith_deg_range,
                azi_step_deg=azi_step_deg,
                zenith_step_deg=zenith_step_deg,
                target_count=target_count,
                seed=seed,
            )

        raise ValueError(f"unknown mode: {mode}")

    @staticmethod
    def _solstice_day(
        *,
        timezone_str: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
        start_dt: Optional[datetime],
        hours: int,
        step_hours: int,
    ) -> RayBuildResult:
        if timezone_str is None or lat is None or lon is None:
            raise ValueError("solstice_day需要timezone_str,lat,lon")

        if start_dt is None:
            start_dt = datetime(2026, 6, 21, 6, 0, 0)

        if step_hours <= 0:
            raise ValueError("step_hours必须>0")
        if hours <= 0:
            raise ValueError("hours必须>0")

        n = int((hours + step_hours - 1) // step_hours)
        timetable = [start_dt + timedelta(hours=i * step_hours) for i in range(n)]

        sun_dirs = build_sun_dirs(timezone_str, float(lat), float(lon), timetable)
        if int(sun_dirs.shape[0]) == 0:
            raise RuntimeError("时间表内太阳始终在地平线下")

        sun_dirs = np.asarray(sun_dirs, dtype=np.float32)
        return RayBuildResult(sun_dirs=sun_dirs, timetable=timetable)

    @staticmethod
    def _year_hourly_2026(
        *,
        timezone_str: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
    ) -> RayBuildResult:
        if timezone_str is None or lat is None or lon is None:
            raise ValueError("year_hourly_2026需要timezone_str,lat,lon")

        if start_dt is None:
            start_dt = datetime(2026, 1, 1, 0, 0, 0)
        if end_dt is None:
            end_dt = datetime(2027, 1, 1, 0, 0, 0)

        if end_dt <= start_dt:
            raise ValueError("end_dt必须晚于start_dt")

        hours = int((end_dt - start_dt).total_seconds() // 3600)
        timetable = [start_dt + timedelta(hours=i) for i in range(hours)]

        sun_dirs = build_sun_dirs(timezone_str, float(lat), float(lon), timetable)
        if int(sun_dirs.shape[0]) == 0:
            raise RuntimeError("全年逐时太阳始终在地平线下(不合理,请检查参数)")

        sun_dirs = np.asarray(sun_dirs, dtype=np.float32)
        return RayBuildResult(sun_dirs=sun_dirs, timetable=timetable)

    @staticmethod
    def _sky_view_factor(
        *,
        azi_deg_range: Tuple[float, float],
        zenith_deg_range: Tuple[float, float],
        azi_step_deg: Optional[float],
        zenith_step_deg: Optional[float],
        target_count: int,
        seed: int,
    ) -> np.ndarray:
        azi0, azi1 = float(azi_deg_range[0]), float(azi_deg_range[1])
        z0, z1 = float(zenith_deg_range[0]), float(zenith_deg_range[1])

        if target_count <= 0:
            raise ValueError("target_count必须>0")
        if azi1 <= azi0:
            raise ValueError("azi_deg_range需要(小,大)")
        if z1 <= z0:
            raise ValueError("zenith_deg_range需要(小,大)")

        # 你要求我自己定step:默认给一个接近200条的网格
        # 360/18=20, 90/9=10 => 200
        if azi_step_deg is None:
            azi_step_deg = 18.0
        if zenith_step_deg is None:
            zenith_step_deg = 9.0

        if azi_step_deg <= 0.0 or zenith_step_deg <= 0.0:
            raise ValueError("azi_step_deg和zenith_step_deg必须>0")

        # 生成网格中心点,避免包含边界重复与极点过密
        azi_bins = max(1, int(np.ceil((azi1 - azi0) / azi_step_deg)))
        zen_bins = max(1, int(np.ceil((z1 - z0) / zenith_step_deg)))

        azi_centers = azi0 + (np.arange(azi_bins, dtype=np.float32) + 0.5) * (
            (azi1 - azi0) / azi_bins
        )
        zen_centers = z0 + (np.arange(zen_bins, dtype=np.float32) + 0.5) * (
            (z1 - z0) / zen_bins
        )

        A, Z = np.meshgrid(azi_centers, zen_centers, indexing="xy")
        A = A.reshape(-1)
        Z = Z.reshape(-1)

        # azimuth从北向顺时针,zenith从天顶向下
        a = np.deg2rad(A.astype(np.float32))
        z = np.deg2rad(Z.astype(np.float32))

        sinz = np.sin(z)
        x = np.sin(a) * sinz  # East
        y = np.cos(a) * sinz  # North
        zz = np.cos(z)        # Up

        dirs = np.stack([x, y, zz], axis=1).astype(np.float32)

        # 均匀抽样到约200条(或你指定target_count)
        n = int(dirs.shape[0])
        if n > target_count:
            rng = np.random.RandomState(int(seed))
            idx = rng.choice(n, size=int(target_count), replace=False)
            dirs = dirs[idx]

        # 归一化(数值安全)
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12).astype(np.float32)
        dirs = (dirs / norm).astype(np.float32)

        # 半球:保证z>=0
        dirs = dirs[dirs[:, 2] >= 0.0]
        if int(dirs.shape[0]) == 0:
            raise RuntimeError("sky_view_factor生成结果为空,请检查范围与步长")

        return dirs