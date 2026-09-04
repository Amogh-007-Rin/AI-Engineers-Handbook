"""TorchServe model-archive manifest contract validator."""


def validate_archive(archive):
    required = ("name", "version", "serialized_file", "handler", "runtime", "signature")
    missing = [key for key in required if not archive.get(key)]
    if missing:
        raise ValueError("archive missing: " + ", ".join(missing))
    if archive["version"] == "latest" or archive.get("management_public"):
        raise ValueError("archive version must be immutable and management private")
    return True
