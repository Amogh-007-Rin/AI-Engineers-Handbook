"""Dependency-free MLflow run manifest and promotion policy."""

import math
import os
from pathlib import Path


REQUIRED = ("run_id", "git_sha", "data_hash", "environment", "metrics", "artifacts")


def validate_run(run):
    missing = [key for key in REQUIRED if key not in run]
    if missing:
        raise ValueError("missing run evidence: " + ", ".join(missing))
    if not run["run_id"] or len(run["git_sha"]) < 7 or len(run["data_hash"]) < 8:
        raise ValueError("run and provenance identifiers are invalid")
    if not run["environment"] or not isinstance(run["metrics"], dict):
        raise ValueError("environment and metrics are required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in run["metrics"].values()):
        raise ValueError("metrics must be finite numeric values")
    return True


def can_promote(run, *, minimum_score):
    validate_run(run)
    score = run["metrics"].get("validation_score")
    if score is None or score < minimum_score or not run["artifacts"].get("model_signature"):
        return False
    return True


def record_local_run(directory, evidence):
    """Persist a real MLflow run to a local file store and return its manifest."""
    validate_run({**evidence, "run_id": evidence.get("run_id", "pending-run")})
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{directory / 'tracking.db'}")
    experiment = client.get_experiment_by_name("handbook-offline")
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(
                         "handbook-offline",
                         artifact_location=(directory / "artifacts").as_uri(),
                     ))
    run = client.create_run(experiment_id, tags={
        "git_sha": evidence["git_sha"], "data_hash": evidence["data_hash"],
        "environment": evidence["environment"],
    })
    run_id = run.info.run_id
    for name, value in evidence["metrics"].items():
        client.log_metric(run_id, name, value)
    client.log_text(run_id, '{"inputs":"double","outputs":"double"}',
                    "model/signature.json")
    client.set_terminated(run_id, status="FINISHED")

    stored = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id, "model")
    return {
        "run_id": run_id,
        "git_sha": stored.data.tags["git_sha"],
        "data_hash": stored.data.tags["data_hash"],
        "environment": stored.data.tags["environment"],
        "metrics": dict(stored.data.metrics),
        "artifacts": {"model_signature": next(
            (item.path for item in artifacts if item.path.endswith("signature.json")), None)},
    }
