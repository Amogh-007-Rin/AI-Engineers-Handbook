"""Deterministic Optuna fixture demonstrating storage and conditional spaces."""

from __future__ import annotations
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial) -> float:
    family = trial.suggest_categorical("family", ["quadratic", "offset"])
    x = trial.suggest_float("x", -5.0, 5.0)
    penalty = trial.suggest_float("penalty", 0.0, 1.0) if family == "offset" else 0.0
    trial.set_user_attr("objective_version", "v1")
    return (x - 2.0) ** 2 + penalty


def run_study(trials: int = 30) -> optuna.Study:
    if trials <= 0:
        raise ValueError("trials must be positive")
    sampler = optuna.samplers.RandomSampler(seed=7)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, catch=(ArithmeticError,))
    return study
