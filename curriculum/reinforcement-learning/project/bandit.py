"""Reproducible epsilon-greedy bandit components."""

from __future__ import annotations

import random


class EpsilonGreedy:
    def __init__(self, arms: int, epsilon: float, seed: int = 0) -> None:
        if arms <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("arms must be positive and epsilon in [0, 1]")
        self.epsilon = epsilon
        self.counts = [0] * arms
        self.values = [0.0] * arms
        self.random = random.Random(seed)

    def choose(self) -> int:
        if self.random.random() < self.epsilon:
            return self.random.randrange(len(self.values))
        best = max(self.values)
        return next(index for index, value in enumerate(self.values) if value == best)

    def update(self, arm: int, reward: float) -> None:
        if arm not in range(len(self.values)) or reward not in (0.0, 1.0, 0, 1):
            raise ValueError("invalid arm or non-Bernoulli reward")
        self.counts[arm] += 1
        self.values[arm] += (float(reward) - self.values[arm]) / self.counts[arm]


def reward_table(probabilities: list[float], rounds: int, seed: int) -> list[list[int]]:
    if rounds < 0 or not probabilities or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probabilities or rounds")
    rng = random.Random(seed)
    return [[int(rng.random() < p) for p in probabilities] for _ in range(rounds)]
