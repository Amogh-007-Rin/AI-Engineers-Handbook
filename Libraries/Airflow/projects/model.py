"""Dependency-free Airflow DAG specification validator."""


def validate_dag(spec):
    tasks = spec.get("tasks", {})
    if not spec.get("dag_id") or not spec.get("owner") or not tasks:
        raise ValueError("dag_id, owner, and tasks are required")
    if spec.get("timezone") is None or spec.get("schedule") is None:
        raise ValueError("schedule and timezone must be explicit")
    for name, task in tasks.items():
        if task.get("retries", -1) < 0 or task.get("timeout_s", 0) <= 0:
            raise ValueError(f"bounded retries and timeout required for {name}")
        if task.get("side_effects") and not task.get("idempotent"):
            raise ValueError(f"side-effecting task {name} must be idempotent")
    dependencies = {(left, right) for left, right in spec.get("dependencies", [])}
    if any(left not in tasks or right not in tasks or left == right for left, right in dependencies):
        raise ValueError("invalid task dependency")
    return True
