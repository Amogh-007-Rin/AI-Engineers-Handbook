"""Tiny deterministic Gymnasium environment contract."""

import gymnasium as gym
from gymnasium import spaces


class LineWorld(gym.Env):
    observation_space = spaces.Discrete(5)
    action_space = spaces.Discrete(2)

    def __init__(self, horizon=8):
        self.horizon = horizon
        self.position = 0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.steps = 0, 0
        return self.position, {"seeded": seed is not None}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action outside declared space")
        self.position += 1 if action else -1
        self.position = max(0, min(4, self.position)); self.steps += 1
        terminated = self.position == 4
        truncated = self.steps >= self.horizon and not terminated
        return self.position, float(terminated), terminated, truncated, {}
