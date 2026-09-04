"""Deterministic, headless Matplotlib report used for artifact testing."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-engineers-handbook-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def draw_series(ax: Axes, x: list[float], y: list[float], lower: list[float], upper: list[float]):
    lengths = {len(x), len(y), len(lower), len(upper)}
    if len(lengths) != 1 or not x:
        raise ValueError("all series must have equal nonzero length")
    line, = ax.plot(x, y, marker="o", label="estimate")
    band = ax.fill_between(x, lower, upper, alpha=0.25, label="uncertainty")
    ax.set(xlabel="Time (day)", ylabel="Rate (%)", title="Rate over time")
    ax.legend()
    ax.grid(True, alpha=0.2)
    return line, band


def build_report() -> tuple[Figure, Axes]:
    with plt.rc_context({"figure.figsize": (6, 4), "font.size": 10}):
        fig, ax = plt.subplots(layout="constrained")
        draw_series(ax, [1, 2, 3], [20, 24, 23], [18, 21, 20], [22, 27, 26])
        return fig, ax


def save_report(path: Path) -> None:
    fig, _ = build_report()
    try:
        fig.savefig(path, dpi=100, metadata={"Software": "AI Engineers Handbook"})
    finally:
        plt.close(fig)
