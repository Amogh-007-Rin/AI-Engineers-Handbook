"""Dependency-free W&B-style run config sanitizer and sweep validator."""

import os


SECRET_WORDS = ("token", "password", "secret", "api_key")


def sanitize(config):
    return {key: "[REDACTED]" if any(word in key.casefold() for word in SECRET_WORDS) else value
            for key, value in config.items()}


def validate_sweep(sweep):
    if sweep.get("method") not in {"grid", "random", "bayes"}:
        raise ValueError("unsupported sweep method")
    if sweep.get("count", 0) < 1 or sweep.get("metric", {}).get("name") is None:
        raise ValueError("sweep requires positive budget and metric")
    return True
