"""Docker Compose-style service contract validator (no daemon required)."""


def validate_service(service):
    required = ("image", "command", "healthcheck", "user", "resources")
    missing = [key for key in required if key not in service]
    if missing:
        raise ValueError(f"missing service contract: {', '.join(missing)}")
    image = service["image"]
    if ":latest" in image or "@sha256:" not in image:
        raise ValueError("image must use an immutable digest")
    if str(service["user"]) in {"0", "root"}:
        raise ValueError("service must not run as root")
    for limit in ("cpus", "memory"):
        if limit not in service["resources"]:
            raise ValueError(f"resource limit missing: {limit}")
    return True
