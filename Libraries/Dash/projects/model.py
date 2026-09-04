"""Small Dash application with directly testable callback logic."""

from dash import Dash, Input, Output, dcc, html


def summarize(values):
    if not isinstance(values, list) or not values:
        raise ValueError("values must be a non-empty list")
    if not all(isinstance(value, (int, float)) for value in values):
        raise TypeError("values must be numeric")
    return sum(values) / len(values)


def create_app():
    app = Dash(__name__)
    app.layout = html.Main([dcc.Input(id="values", value="1,2,3"), html.Output(id="result")])

    @app.callback(Output("result", "children"), Input("values", "value"))
    def update(raw):
        try:
            values = [float(item.strip()) for item in raw.split(",")]
            return f"Mean: {summarize(values):.2f}"
        except (AttributeError, TypeError, ValueError):
            return "Enter comma-separated numbers"
    return app
