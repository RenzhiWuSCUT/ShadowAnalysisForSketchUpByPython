# exportSU_util.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import ctypes as ct

import numpy as np

import obj._SU_API as SU


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v * 0.0
    return v / n


def project_to_uv(
    basis_p0: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    pts3: np.ndarray,
) -> np.ndarray:
    d = pts3 - basis_p0[None, :]
    uu = d @ basis_u
    vv = d @ basis_v
    return np.stack([uu, vv], axis=1).astype(np.float64)


def signed_area_2d(poly2: np.ndarray) -> float:
    if poly2.shape[0] < 3:
        return 0.0
    x = poly2[:, 0].astype(np.float64, copy=False)
    y = poly2[:, 1].astype(np.float64, copy=False)
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - x2 * y))


def ensure_ccw_in_basis(
    verts: List[Tuple[float, float, float]],
    basis_p0: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> List[Tuple[float, float, float]]:
    P = np.asarray(verts, dtype=np.float64)
    uv = project_to_uv(basis_p0, basis_u, basis_v, P)
    if signed_area_2d(uv) < 0.0:
        return list(reversed(verts))
    return verts


def ensure_cw_in_basis(
    verts: List[Tuple[float, float, float]],
    basis_p0: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> List[Tuple[float, float, float]]:
    P = np.asarray(verts, dtype=np.float64)
    uv = project_to_uv(basis_p0, basis_u, basis_v, P)
    if signed_area_2d(uv) > 0.0:
        return list(reversed(verts))
    return verts


def bounds_uv_from_verts(
    verts: List[Tuple[float, float, float]],
    basis_p0: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    P = np.asarray(verts, dtype=np.float64)
    if P.shape[0] < 3:
        return 0.0, 1.0, 0.0, 1.0, False
    uv = project_to_uv(basis_p0, basis_u, basis_v, P)
    u_min = float(np.min(uv[:, 0]))
    u_max = float(np.max(uv[:, 0]))
    v_min = float(np.min(uv[:, 1]))
    v_max = float(np.max(uv[:, 1]))
    if (u_max - u_min) <= 1e-12 or (v_max - v_min) <= 1e-12:
        return u_min, u_max, v_min, v_max, False
    return u_min, u_max, v_min, v_max, True


def get_geo4_bounds_for_face(
    geo4: Any,
    fi: int,
) -> tuple[Optional[Tuple[float, float, float, float]], bool]:
    if geo4 is None:
        return None, False
    if not hasattr(geo4, "faceBoundUVL") or not hasattr(geo4, "faceBoundValidL"):
        return None, False

    b = getattr(geo4, "faceBoundUVL")
    v = getattr(geo4, "faceBoundValidL")
    if b is None or v is None:
        return None, False

    vv = np.asarray(v)
    if int(fi) < 0 or int(fi) >= int(vv.shape[0]):
        return None, False

    if int(vv[int(fi)]) != 1:
        return None, False

    row = np.asarray(b)[int(fi)]
    if row.shape[0] < 4:
        return None, False

    u_min = float(row[0])
    u_max = float(row[1])
    v_min = float(row[2])
    v_max = float(row[3])
    if (u_max - u_min) <= 1e-12 or (v_max - v_min) <= 1e-12:
        return None, False

    return (u_min, u_max, v_min, v_max), True


def bound_corners_3d(
    basis_p0: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> List[Tuple[float, float, float]]:
    c00 = basis_p0 + basis_u * u_min + basis_v * v_min
    c10 = basis_p0 + basis_u * u_max + basis_v * v_min
    c11 = basis_p0 + basis_u * u_max + basis_v * v_max
    c01 = basis_p0 + basis_u * u_min + basis_v * v_max
    return [
        (float(c00[0]), float(c00[1]), float(c00[2])),
        (float(c10[0]), float(c10[1]), float(c10[2])),
        (float(c11[0]), float(c11[1]), float(c11[2])),
        (float(c01[0]), float(c01[1]), float(c01[2])),
    ]


@dataclass
class SketchUpAPI:
    dll_path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(self.dll_path)

        self.dll = ct.WinDLL(self.dll_path)

        SU.bind_export_api(self.dll)
        SU.bind_core_api(self.dll)