from __future__ import annotations

import os
from typing import Dict, Tuple, List

import numpy as np

from obj.Geo import SketchUpGeo, Texture, Material, UV, Point3
from util.geo_sampling import GeoSampling
from util.img_save import ensure_dir, scalar_to_rgba_u8, save_rgba_png


def _project_to_uv(geo: SketchUpGeo, fi: int, pts3: np.ndarray) -> np.ndarray:
    basis = GeoSampling.face_plane_basis(geo, int(fi))
    p0 = basis.p0.astype(np.float64, copy=False)
    u = basis.u.astype(np.float64, copy=False)
    v = basis.v.astype(np.float64, copy=False)
    d = pts3.astype(np.float64, copy=False) - p0[None, :]
    uu = d @ u
    vv = d @ v
    return np.stack([uu, vv], axis=1).astype(np.float64)


def _fill_linear_1d_rowwise(img: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 横向线性填充
    H, W = img.shape
    out = img.copy()
    vout = valid.copy()

    xs = np.arange(W, dtype=np.int32)
    for y in range(H):
        m = vout[y]
        if not np.any(m):
            continue
        idx = xs[m]
        val = out[y, m]
        if idx.shape[0] == 1:
            out[y, :] = val[0]
            vout[y, :] = True
            continue
        out[y, :] = np.interp(xs.astype(np.float32), idx.astype(np.float32), val.astype(np.float32)).astype(np.float32)
        vout[y, :] = True
    return out, vout


def _fill_linear_1d_colwise(img: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 纵向线性填充
    H, W = img.shape
    out = img.copy()
    vout = valid.copy()

    ys = np.arange(H, dtype=np.int32)
    for x in range(W):
        m = vout[:, x]
        if not np.any(m):
            continue
        idx = ys[m]
        val = out[m, x]
        if idx.shape[0] == 1:
            out[:, x] = val[0]
            vout[:, x] = True
            continue
        out[:, x] = np.interp(ys.astype(np.float32), idx.astype(np.float32), val.astype(np.float32)).astype(np.float32)
        vout[:, x] = True
    return out, vout


def _bake_one_face_bilinear(
    geo: SketchUpGeo,
    fi: int,
    pts3: np.ndarray,
    val: np.ndarray,
    faceBoundUV: np.ndarray,
    *,
    tex_size: int,
) -> np.ndarray:
    umin, umax, vmin, vmax = [float(x) for x in faceBoundUV]
    W = int(tex_size)
    H = int(tex_size)
    du = umax - umin
    dv = vmax - vmin
    if du <= 0.0:
        du = 1e-6
    if dv <= 0.0:
        dv = 1e-6

    uv = _project_to_uv(geo, fi, pts3)

    fx = (uv[:, 0] - umin) / du * (W - 1)
    fy = (uv[:, 1] - vmin) / dv * (H - 1)

    good = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(val)
    fx = fx[good]
    fy = fy[good]
    val = val[good]

    ix = np.clip(np.rint(fx), 0, W - 1).astype(np.int32)
    iy = np.clip(np.rint(fy), 0, H - 1).astype(np.int32)

    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.int32)

    np.add.at(acc, (iy, ix), val.astype(np.float32, copy=False))
    np.add.at(cnt, (iy, ix), 1)

    valid = cnt > 0
    img = np.zeros((H, W), dtype=np.float32)
    img[valid] = acc[valid] / np.maximum(cnt[valid], 1).astype(np.float32)

    img, valid = _fill_linear_1d_rowwise(img, valid)
    img, valid = _fill_linear_1d_colwise(img, valid)

    return img

def bake_faces_bilinear(
    geo: SketchUpGeo,
    sample_pts: np.ndarray,
    sample_face: np.ndarray,
    sample_on: np.ndarray,
    sample_val: np.ndarray,
    faceBoundUVL: np.ndarray,
    faceBoundValidL: np.ndarray,
    *,
    tex_size: int = 256,
    out_dir: str = "_temp_sunshade_interp",
    clamp: Tuple[float, float] = (0.0, 1.0),
    cmap_name: str = "turbo",
) -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
    ensure_dir(out_dir)

    pts = sample_pts.astype(np.float64, copy=False)
    fid = sample_face.astype(np.int32, copy=False)
    on = sample_on.astype(np.int32, copy=False)
    val = sample_val.astype(np.float32, copy=False)

    F = int(len(geo.faces))
    face_to_png: Dict[int, str] = {}
    face_to_scalar: Dict[int, np.ndarray] = {}

    for fi in range(F):
        if int(faceBoundValidL[fi]) == 0:
            continue

        m = (fid == fi) & (on > 0)
        if not np.any(m):
            continue

        img_scalar = _bake_one_face_bilinear(
            geo,
            fi,
            pts[m],
            val[m],
            faceBoundUVL[fi],
            tex_size=tex_size,
        )

        rgba = scalar_to_rgba_u8(img_scalar, clamp=clamp, cmap_name=cmap_name)
        png_path = os.path.join(out_dir, f"face_{fi:06d}.png")
        save_rgba_png(png_path, rgba)

        face_to_png[fi] = png_path
        face_to_scalar[fi] = img_scalar

    return face_to_png, face_to_scalar


def attach_baked_textures_to_geo(
    geo: SketchUpGeo,
    face_to_png: Dict[int, str],
    faceBoundUVL: np.ndarray,
) -> None:
    for fi, png_path in face_to_png.items():
        fi = int(fi)
        face = geo.faces[fi]

        umin, umax, vmin, vmax = [float(x) for x in faceBoundUVL[fi]]
        w = umax - umin
        h = vmax - vmin
        if w <= 0.0:
            w = 1e-6
        if h <= 0.0:
            h = 1e-6

        basis = GeoSampling.face_plane_basis(geo, fi)
        base_p = Point3(float(basis.p0[0]), float(basis.p0[1]), float(basis.p0[2]))

        loop_idx_list: List[int] = [int(face.outer_loop)] + [int(x) for x in face.inner_loops]

        vert_idx_all: List[int] = []
        for li in loop_idx_list:
            lp = geo.loops[li]
            if len(lp.verts) >= 3:
                vert_idx_all.extend([int(v) for v in lp.verts])

        if len(vert_idx_all) < 3:
            continue

        pts3 = np.array(
            [[float(geo.points[vi].x), float(geo.points[vi].y), float(geo.points[vi].z)] for vi in vert_idx_all],
            dtype=np.float64,
        )
        uv2 = _project_to_uv(geo, fi, pts3)

        uv_list: List[UV] = []
        for k in range(uv2.shape[0]):
            uu = (float(uv2[k, 0]) - umin) / w
            vv = (float(uv2[k, 1]) - vmin) / h
            uv_list.append(UV(float(uu), float(vv)))

        tex = Texture(
            png=str(png_path),
            uv=uv_list,
            base_point=base_p,
            width=float(w),
            height_v=float(h),
        )
        tex_idx = len(geo.textures)
        geo.textures.append(tex)

        mat = Material(name=f"sunshade_{fi:06d}")
        mat.texture_index = int(tex_idx)
        mat_idx = len(geo.materials)
        geo.materials.append(mat)

        face.front_material = int(mat_idx)
        face.back_material = int(mat_idx)