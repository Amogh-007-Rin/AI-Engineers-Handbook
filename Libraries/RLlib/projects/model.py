"""Dependency-free RLlib experiment configuration validator."""


def validate_experiment(config):
    if not config.get("algorithm") or not config.get("environment"):
        raise ValueError("algorithm and environment required")
    if config.get("environment_runners", -1) < 0 or config.get("cpus_per_runner", 0) <= 0:
        raise ValueError("runner and resource configuration invalid")
    evaluation = config.get("evaluation", {})
    if evaluation.get("episodes", 0) < 1 or evaluation.get("explore", True):
        raise ValueError("fixed non-exploratory evaluation required")
    if config.get("checkpoint_interval", 0) < 1:
        raise ValueError("checkpoint interval required")
    return True
