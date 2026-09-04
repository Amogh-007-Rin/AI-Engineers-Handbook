"""Stable-Baselines3 experiment contract validator."""


SUPPORTED = {"discrete": {"PPO", "A2C", "DQN"}, "continuous": {"PPO", "A2C", "SAC", "TD3"}}


def validate_experiment(spec):
    space, algorithm = spec.get("action_space"), spec.get("algorithm")
    if algorithm not in SUPPORTED.get(space, set()):
        raise ValueError("algorithm incompatible with action space")
    if len(spec.get("seeds", [])) < 3 or spec.get("evaluation_episodes", 0) < 5:
        raise ValueError("multi-seed evaluation with at least five episodes required")
    if spec.get("normalized") and not spec.get("normalization_frozen_for_eval"):
        raise ValueError("evaluation normalization must be frozen")
    return True
