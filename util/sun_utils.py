from __future__ import annotations
import math
from datetime import datetime
from typing import List
import numpy as np
import pvlib
import pytz

_UNIT_TO_M = {
    "Inches": 0.0254,
    "Feet": 0.3048,
    "Millimeters": 0.001,
    "Centimeters": 0.01,
    "Meters": 1.0,
}


def sun_dir_from_pvlib(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    ze = math.radians(float(zenith_deg))
    x = math.sin(ze) * math.sin(az)  # east
    y = math.sin(ze) * math.cos(az)  # north
    z = math.cos(ze)                 # up
    v = np.array([x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return v / n


def build_sun_dirs(timezone_str: str, lat: float, lon: float, times: List[datetime]) -> np.ndarray:
    tz = pytz.timezone(timezone_str)
    localized: List[datetime] = []
    for t in times:
        if t.tzinfo is None:
            localized.append(tz.localize(t))
        else:
            localized.append(t.astimezone(tz))

    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz=timezone_str)
    sp = loc.get_solarposition(localized)

    dirs: List[np.ndarray] = []
    for az, ze in zip(sp["azimuth"].values, sp["apparent_zenith"].values):
        d = sun_dir_from_pvlib(float(az), float(ze))
        if d[2] > 0.0:
            dirs.append(d)

    if len(dirs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(dirs, axis=0).astype(np.float32)