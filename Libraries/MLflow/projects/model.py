"""Dependency-free MLflow run manifest and promotion policy."""


REQUIRED = ("run_id", "git_sha", "data_hash", "environment", "metrics", "artifacts")


def validate_run(run):
    missing = [key for key in REQUIRED if key not in run]
    if missing:
        raise ValueError("missing run evidence: " + ", ".join(missing))
    if not run["run_id"] or len(run["git_sha"]) < 7 or len(run["data_hash"]) < 8:
        raise ValueError("run and provenance identifiers are invalid")
    if any(value is None for value in run["metrics"].values()):
        raise ValueError("metrics must be finite values")
    return True


def can_promote(run, *, minimum_score):
    validate_run(run)
    score = run["metrics"].get("validation_score")
    if score is None or score < minimum_score or not run["artifacts"].get("model_signature"):
        return False
    return True
