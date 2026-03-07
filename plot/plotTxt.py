# plot/plotTxt.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


@dataclass
class PlotTxt:
    @staticmethod
    def draw_rotated_text(
        ax,
        *,
        x: float,
        y: float,
        text: str,
        rotation_deg: float = 90.0,
        fontsize: int = 10,
        color: str = "white",
        stroke_color: str = "black",
        stroke_width: float = 2.0,
    ) -> None:
        """
        在给定matplotlib axes上画旋转文字（默认竖排效果：rotation=90）。
        """
        if text is None:
            return
        t = ax.text(
            float(x),
            float(y),
            str(text),
            rotation=float(rotation_deg),
            ha="center",
            va="center",
            fontsize=int(fontsize),
            color=str(color),
        )
        t.set_path_effects(
            [pe.Stroke(linewidth=float(stroke_width), foreground=str(stroke_color)), pe.Normal()]
        )

    @staticmethod
    def save_text_png(
        out_png: str,
        *,
        text: str,
        size_px: Tuple[int, int] = (256, 256),
        dpi: int = 200,
        rotation_deg: float = 90.0,
        fontsize: int = 18,
        color: str = "white",
        stroke_color: str = "black",
        stroke_width: float = 2.0,
    ) -> None:
        """
        单独输出一个透明背景文字PNG（需要时可以贴到面上）。
        """
        _ensure_dir(out_png)
        W = int(size_px[0])
        H = int(size_px[1])
        if W <= 0 or H <= 0:
            raise ValueError("size_px必须>0")

        fig_w = float(W) / float(dpi)
        fig_h = float(H) / float(dpi)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        PlotTxt.draw_rotated_text(
            ax,
            x=0.5,
            y=0.5,
            text=str(text),
            rotation_deg=float(rotation_deg),
            fontsize=int(fontsize),
            color=str(color),
            stroke_color=str(stroke_color),
            stroke_width=float(stroke_width),
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        fig.savefig(out_png, transparent=True)
        plt.close(fig)