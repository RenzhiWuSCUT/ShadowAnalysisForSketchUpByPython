# plot采样点预览.py
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


class SamplePreview:
    @staticmethod
    def show_points(
        ptL: np.ndarray,
        faceIdxL: np.ndarray,
        isOnFaceL: np.ndarray,
        *,
        title: str = "采样点预览",
        point_size: float = 1.0,
    ) -> None:
        """
        输入：
        ptL: (N,3)
        faceIdxL: (N,)
        isOnFaceL: (N,) 0/1
        效果：
        3D散点；按faceIdxL着色；isOnFaceL=0透明度40%(alpha=0.4)
        Plotly版：两个trace实现不同透明度，颜色按faceIdxL映射。
        """
        if ptL.ndim != 2 or ptL.shape[1] != 3:
            raise ValueError("ptL must be (N,3)")
        if faceIdxL.ndim != 1 or faceIdxL.shape[0] != ptL.shape[0]:
            raise ValueError("faceIdxL must be (N,)")
        if isOnFaceL.ndim != 1 or isOnFaceL.shape[0] != ptL.shape[0]:
            raise ValueError("isOnFaceL must be (N,)")

        pts = ptL.astype(np.float64, copy=False)
        fids = faceIdxL.astype(np.int32, copy=False)
        on = isOnFaceL.astype(np.int32, copy=False)

        # faceIdx归一化成0..1，用连续色带
        fid_min = int(np.min(fids)) if fids.size else 0
        fid_max = int(np.max(fids)) if fids.size else 1
        denom = float(max(1, fid_max - fid_min))
        t = (fids.astype(np.float64) - float(fid_min)) / denom

        mask_on = on > 0
        mask_off = ~mask_on

        fig = go.Figure()

        if np.any(mask_on):
            fig.add_trace(
                go.Scatter3d(
                    x=pts[mask_on, 0],
                    y=pts[mask_on, 1],
                    z=pts[mask_on, 2],
                    mode="markers",
                    name="onFace",
                    marker=dict(
                        size=float(point_size),
                        color=t[mask_on],
                        colorscale="Turbo",
                        opacity=1.0,
                        showscale=True,
                        colorbar=dict(title="faceIdx(norm)"),
                    ),
                    hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
                )
            )

        if np.any(mask_off):
            fig.add_trace(
                go.Scatter3d(
                    x=pts[mask_off, 0],
                    y=pts[mask_off, 1],
                    z=pts[mask_off, 2],
                    mode="markers",
                    name="offFace",
                    marker=dict(
                        size=float(point_size),
                        color=t[mask_off],
                        colorscale="Turbo",
                        opacity=0.1,
                        showscale=False,
                    ),
                    hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
                )
            )

        # 等比例外观：用bbox推一个中心+半径
        x0, x1 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y0, y1 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        z0, z1 = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        cz = 0.5 * (z0 + z1)
        r = 0.5 * max(x1 - x0, y1 - y0, z1 - z0, 1e-9)

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X", range=[cx - r, cx + r]),
                yaxis=dict(title="Y", range=[cy - r, cy + r]),
                zaxis=dict(title="Z", range=[cz - r, cz + r]),
                aspectmode="cube",
            ),
            legend=dict(itemsizing="constant"),
        )

        fig.show()