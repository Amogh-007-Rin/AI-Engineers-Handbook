"""Plotly figure specification contract."""

from pathlib import Path
import plotly.graph_objects as go


def figure(labels, values):
    if len(labels) != len(values) or not labels:
        raise ValueError("labels and values must be non-empty and equal length")
    return go.Figure(go.Bar(x=list(labels), y=list(values), customdata=[[label] for label in labels],
                            hovertemplate="%{customdata[0]}: %{y}<extra></extra>"))


def export(labels, values, path):
    path = Path(path); figure(labels, values).write_html(path, include_plotlyjs="cdn")
    return path
