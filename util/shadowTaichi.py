# shadowTaichi.py
from __future__ import annotations

import numpy as np
import taichi as ti


@ti.data_oriented
class ShadowTaichi:
    """
    阴影可见度计算(点向太阳方向发射射线，与三角面求交)
    产出ptVis:(P,) float32，范围0..1，表示S条sun_dirs中无遮挡比例

    约束遵守：
    1 不用ti.i32做@ti.func参数注解
    2 不在kernel参数里用ti.template
    3 @ti.func内部不在运行时分支/循环内提前return
    """

    def __init__(self, P: int, T: int, S: int, F: int, *, eps: float = 1e-3):
        self.P = int(P)
        self.T = int(T)
        self.S = int(S)
        self.F = int(F)
        self.eps = float(eps)

        self.pts = ti.Vector.field(3, dtype=ti.f32, shape=self.P)
        self.on = ti.field(dtype=ti.i32, shape=self.P)
        self.p_face = ti.field(dtype=ti.i32, shape=self.P)

        self.face_n = ti.Vector.field(3, dtype=ti.f32, shape=self.F)
        self.dirs = ti.Vector.field(3, dtype=ti.f32, shape=self.S)

        self.tri_v0 = ti.Vector.field(3, dtype=ti.f32, shape=self.T)
        self.tri_v1 = ti.Vector.field(3, dtype=ti.f32, shape=self.T)
        self.tri_v2 = ti.Vector.field(3, dtype=ti.f32, shape=self.T)
        self.tri_face = ti.field(dtype=ti.i32, shape=self.T)  # 每个三角形所属face

        self.vis_sum = ti.field(dtype=ti.f32, shape=self.P)

        self.eps_s = ti.field(dtype=ti.f32, shape=())
        self.eps_s[None] = np.float32(self.eps)

    def set_inputs(
        self,
        pts: np.ndarray,
        is_on: np.ndarray,
        p_face: np.ndarray,
        face_normals: np.ndarray,
        sun_dirs: np.ndarray,
        tri_v0: np.ndarray,
        tri_v1: np.ndarray,
        tri_v2: np.ndarray,
        tri_face: np.ndarray,
    ) -> None:
        if pts.shape[0] != self.P or pts.shape[1] != 3:
            raise ValueError("pts shape mismatch")
        if is_on.shape[0] != self.P:
            raise ValueError("is_on shape mismatch")
        if p_face.shape[0] != self.P:
            raise ValueError("p_face shape mismatch")
        if face_normals.shape[0] != self.F or face_normals.shape[1] != 3:
            raise ValueError("face_normals shape mismatch")
        if sun_dirs.shape[0] != self.S or sun_dirs.shape[1] != 3:
            raise ValueError("sun_dirs shape mismatch")
        if tri_v0.shape[0] != self.T or tri_v0.shape[1] != 3:
            raise ValueError("v0ontriL shape mismatch")
        if tri_v1.shape[0] != self.T or tri_v1.shape[1] != 3:
            raise ValueError("v1ontriL shape mismatch")
        if tri_v2.shape[0] != self.T or tri_v2.shape[1] != 3:
            raise ValueError("v2ontriL shape mismatch")
        if tri_face.shape[0] != self.T:
            raise ValueError("triOnfaceIdxL shape mismatch")

        self.pts.from_numpy(pts.astype(np.float32, copy=False))
        self.on.from_numpy(is_on.astype(np.int32, copy=False))
        self.p_face.from_numpy(p_face.astype(np.int32, copy=False))

        fn = face_normals.astype(np.float32, copy=False)
        nrm = np.linalg.norm(fn, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12).astype(np.float32)
        fn = (fn / nrm).astype(np.float32)
        self.face_n.from_numpy(fn)

        d = sun_dirs.astype(np.float32, copy=False)
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        dn = np.maximum(dn, 1e-12).astype(np.float32)
        d = (d / dn).astype(np.float32)
        self.dirs.from_numpy(d)

        self.tri_v0.from_numpy(tri_v0.astype(np.float32, copy=False))
        self.tri_v1.from_numpy(tri_v1.astype(np.float32, copy=False))
        self.tri_v2.from_numpy(tri_v2.astype(np.float32, copy=False))
        self.tri_face.from_numpy(tri_face.astype(np.int32, copy=False))

    @ti.func
    def _ray_tri_hit(self, o, d, v0, v1, v2, tmin) -> ti.i32:
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = d.cross(e2)
        det = e1.dot(pvec)

        ad = ti.abs(det)
        inv_det = 0.0
        ok_det = 0
        if ad > 1e-10:
            inv_det = 1.0 / det
            ok_det = 1

        tvec = o - v0
        u = tvec.dot(pvec) * inv_det

        qvec = tvec.cross(e1)
        v = d.dot(qvec) * inv_det

        t = e2.dot(qvec) * inv_det

        cond = 1
        if ok_det == 0:
            cond = 0
        if u < 0.0 or u > 1.0:
            cond = 0
        if v < 0.0 or (u + v) > 1.0:
            cond = 0
        if t <= tmin:
            cond = 0

        hit = 0
        if cond == 1:
            hit = 1
        return hit

    @ti.kernel
    def _clear_vis(self):
        for p in range(self.P):
            self.vis_sum[p] = 0.0

    @ti.kernel
    def _accum_vis(self):
        eps0 = self.eps_s[None]
        for p, s in ti.ndrange(self.P, self.S):
            if self.on[p] == 0:
                continue

            fid = self.p_face[p]
            n = self.face_n[fid]
            d = self.dirs[s]

            nd = n.dot(d)
            if nd <= 1e-6:
                continue

            o = self.pts[p] + d * eps0

            occluded = 0
            for t in range(self.T):
                # 关键：同一个face的三角面不参与遮挡判定
                if self.tri_face[t] == fid:
                    continue

                hit = self._ray_tri_hit(o, d, self.tri_v0[t], self.tri_v1[t], self.tri_v2[t], eps0)
                if hit == 1:
                    occluded = 1
            if occluded == 0:
                ti.atomic_add(self.vis_sum[p], 1.0)

    def compute(self) -> np.ndarray:
        self._clear_vis()
        self._accum_vis()
        vis_sum = self.vis_sum.to_numpy().astype(np.float32, copy=False)
        if self.S <= 0:
            raise ValueError("S must be > 0")
        out = vis_sum / float(self.S)
        return out.astype(np.float32, copy=False)


def analyze_shadow_visibility(
    pts: np.ndarray,
    face_idx: np.ndarray,
    is_on: np.ndarray,
    face_normals: np.ndarray,
    sun_dirs: np.ndarray,
    v0ontriL: np.ndarray,
    v1ontriL: np.ndarray,
    v2ontriL: np.ndarray,
    triOnfaceIdxL: np.ndarray,
    *,
    eps: float = 1e-3,
    ti_arch=ti.cuda,
) -> np.ndarray:
    P = int(pts.shape[0])
    T = int(v0ontriL.shape[0])
    S = int(sun_dirs.shape[0])
    F = int(face_normals.shape[0])

    ti.init(arch=ti_arch)

    eng = ShadowTaichi(P, T, S, F, eps=eps)
    eng.set_inputs(
        pts=pts,
        is_on=is_on,
        p_face=face_idx,
        face_normals=face_normals,
        sun_dirs=sun_dirs,
        tri_v0=v0ontriL,
        tri_v1=v1ontriL,
        tri_v2=v2ontriL,
        tri_face=triOnfaceIdxL,
    )
    ptVis = eng.compute()

    off = (is_on.astype(np.int32, copy=False) == 0)
    if np.any(off):
        ptVis = ptVis.copy()
        ptVis[off] = 0.0

    del eng
    return ptVis.astype(np.float32, copy=False)