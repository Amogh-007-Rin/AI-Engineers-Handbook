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


def train_smoke_agent(*, seed=7, timesteps=32):
    """Train a tiny deterministic-policy smoke fixture without external data."""
    if not isinstance(seed, int) or timesteps < 16:
        raise ValueError("seed must be an integer and timesteps must be at least 16")

    # Lazy import preserves the dependency-free specification checks while the
    # academy environment exercises a real framework lifecycle.
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy", "CartPole-v1", n_steps=16, batch_size=16, n_epochs=1,
        seed=seed, device="cpu", verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    observation = model.get_env().reset()
    action, _ = model.predict(observation, deterministic=True)
    if not model.action_space.contains(action[0]):
        raise RuntimeError("trained policy produced an invalid action")
    model.get_env().close()
    return int(action[0])
