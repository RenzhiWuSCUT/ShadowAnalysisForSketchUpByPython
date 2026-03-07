from typing import List, Tuple
import numpy as np
def _point_plane_signed_distance(p: Tuple[float, float, float], plane) -> float:
    # plane: ax + by + cz + d = 0
    a = float(plane.a)
    b = float(plane.b)
    c = float(plane.c)
    d = float(plane.d)
    x = float(p[0])
    y = float(p[1])
    z = float(p[2])
    denom = (a * a + b * b + c * c) ** 0.5
    if denom <= 1e-12:
        return 0.0
    return (a * x + b * y + c * z + d) / denom

def _point_on_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> bool:
    ab = b - a
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    if abs(float(cross)) > eps:
        return False
    dot = float(ap[0] * ab[0] + ap[1] * ab[1])
    if dot < -eps:
        return False
    ab2 = float(ab[0] * ab[0] + ab[1] * ab[1])
    if dot > ab2 + eps:
        return False
    return True


def _point_in_poly_2d(p: np.ndarray, poly: np.ndarray, eps: float = 1e-8) -> bool:
    n = int(poly.shape[0])
    inside = False

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if _point_on_segment_2d(p, a, b, eps=eps):
            return True

        yi = float(a[1])
        yj = float(b[1])
        xi = float(a[0])
        xj = float(b[0])
        py = float(p[1])
        px = float(p[0])

        hit = ((yi > py) != (yj > py))
        if hit:
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside

    return inside


def point_in_face_uv(geo, fi: int, uv: np.ndarray, eps: float = 1e-8) -> bool:
    face = geo.faces[int(fi)]

    outer = np.asarray(face.loops[0].pts, dtype=np.float64)
    if not _point_in_poly_2d(uv, outer, eps=eps):
        return False

    for k in range(1, len(face.loops)):
        hole = np.asarray(face.loops[k].pts, dtype=np.float64)
        if _point_in_poly_2d(uv, hole, eps=eps):
            return False

    return True