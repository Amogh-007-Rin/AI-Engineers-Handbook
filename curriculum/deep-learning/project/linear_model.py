"""A transparent scalar linear model used to verify gradient reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LinearModel:
    weight: float = 0.0
    bias: float = 0.0

    def predict(self, x: float) -> float:
        return self.weight * x + self.bias


def loss_and_gradients(model: LinearModel, xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if not xs or len(xs) != len(ys):
        raise ValueError("xs and ys must have the same nonzero length")
    errors = [model.predict(x) - y for x, y in zip(xs, ys, strict=True)]
    count = len(xs)
    loss = sum(error * error for error in errors) / count
    grad_weight = 2 * sum(error * x for error, x in zip(errors, xs, strict=True)) / count
    grad_bias = 2 * sum(errors) / count
    if not all(math.isfinite(value) for value in (loss, grad_weight, grad_bias)):
        raise FloatingPointError("nonfinite loss or gradient")
    return loss, grad_weight, grad_bias


def train(model: LinearModel, xs: list[float], ys: list[float], learning_rate: float, steps: int) -> list[float]:
    if learning_rate <= 0 or steps < 0:
        raise ValueError("learning_rate must be positive and steps nonnegative")
    history: list[float] = []
    for _ in range(steps):
        loss, grad_weight, grad_bias = loss_and_gradients(model, xs, ys)
        history.append(loss)
        model.weight -= learning_rate * grad_weight
        model.bias -= learning_rate * grad_bias
        if not math.isfinite(model.weight) or not math.isfinite(model.bias):
            raise FloatingPointError("training diverged")
    return history
