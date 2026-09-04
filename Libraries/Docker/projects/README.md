# Docker service contract project

Validate an ML service manifest without requiring a Docker daemon. Run `python
-W error -m unittest -v`; extend it with a real multi-stage build, SBOM, scan,
non-root runtime, graceful shutdown, and digest-pinned integration run.
