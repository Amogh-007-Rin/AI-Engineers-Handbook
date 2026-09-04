"""Dependency-free W&B-style run config sanitizer and sweep validator."""

import math
import os
from pathlib import Path


SECRET_WORDS = ("token", "password", "secret", "api_key")


def sanitize(config):
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")

    def clean(value):
        if isinstance(value, dict):
            return {key: "[REDACTED]" if any(word in str(key).casefold() for word in SECRET_WORDS)
                    else clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(config)


def validate_sweep(sweep):
    if sweep.get("method") not in {"grid", "random", "bayes"}:
        raise ValueError("unsupported sweep method")
    if sweep.get("count", 0) < 1 or sweep.get("metric", {}).get("name") is None:
        raise ValueError("sweep requires positive budget and metric")
    return True


def record_offline_run(directory, config, *, validation_score):
    """Log one real W&B run and artifact without contacting the hosted service."""
    if isinstance(validation_score, bool) or not isinstance(validation_score, (int, float)) \
            or not math.isfinite(validation_score):
        raise ValueError("validation_score must be finite")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in {
        "WANDB_DIR": directory,
        "WANDB_CACHE_DIR": directory / "cache",
        "WANDB_CONFIG_DIR": directory / "config",
        "WANDB_DATA_DIR": directory / "data",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"

    import wandb

    card = directory / "model-card.json"
    card.write_text('{"purpose":"offline academy fixture"}\n', encoding="utf-8")
    run = wandb.init(
        project="handbook-offline",
        config=sanitize(config),
        settings=wandb.Settings(disable_code=True, disable_git=True, silent=True,
                                x_disable_meta=True, x_disable_stats=True),
    )
    try:
        run.log({"validation_score": validation_score}, step=0)
        artifact = wandb.Artifact("handbook-model", type="model",
                                  metadata={"validation_score": validation_score})
        artifact.add_file(str(card))
        run.log_artifact(artifact)
        return {"run_id": run.id, "config": dict(run.config), "run_dir": run.dir}
    finally:
        run.finish()
