"""Bokeh chart contract with deterministic local resources."""

from pathlib import Path
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import CDN


def chart(x, y):
    if len(x) != len(y) or not x:
        raise ValueError("x and y must be non-empty and equal length")
    source = ColumnDataSource({"x": list(x), "y": list(y)})
    plot = figure(title="Measured values", x_axis_label="sample", y_axis_label="value", width=500, height=300)
    plot.line("x", "y", source=source, line_width=2)
    return plot


def export(x, y, path):
    path = Path(path)
    path.write_text(file_html(chart(x, y), CDN, "Measured values"), encoding="utf-8")
    return path
