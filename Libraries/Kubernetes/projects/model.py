"""Kubernetes deployment manifest contract validator (no cluster required)."""


def validate_deployment(manifest):
    spec = manifest.get("spec", {}); template = spec.get("template", {})
    pod = template.get("spec", {}); containers = pod.get("containers", [])
    if not containers:
        raise ValueError("at least one container required")
    for container in containers:
        image = container.get("image", "")
        if "@sha256:" not in image:
            raise ValueError("container image must be digest pinned")
        resources = container.get("resources", {})
        if not resources.get("requests") or not resources.get("limits"):
            raise ValueError("requests and limits are required")
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            if probe not in container:
                raise ValueError(f"missing {probe}")
    return True
