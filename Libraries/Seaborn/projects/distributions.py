"""Headless Seaborn distribution report with semantic contracts."""

from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-engineers-handbook-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def example_data() -> pd.DataFrame:
    return pd.DataFrame({
        "group": pd.Categorical(["control"] * 4 + ["treatment"] * 4, categories=["control", "treatment"], ordered=True),
        "score": [2, 3, 3, 4, 3, 5, 6, 6],
        "subject": [1, 2, 3, 4, 5, 6, 7, 8],
    })


def build_report(data: pd.DataFrame):
    required = {"group", "score", "subject"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("nonempty tidy data with group, score, and subject is required")
    fig, ax = plt.subplots(layout="constrained")
    sns.boxplot(data=data, x="group", y="score", color="white", ax=ax)
    sns.stripplot(data=data, x="group", y="score", color="black", jitter=False, marker="o", ax=ax)
    ax.set(xlabel="Study group", ylabel="Score (points)", title="Score distribution by group")
    return fig, ax


def save_report(path: Path) -> None:
    fig, _ = build_report(example_data())
    try:
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
