---
title: Docker images containers security and reproducible ML builds
slug: docker-foundations
level: practitioner
stage: ml-systems
estimated_hours: 12
prerequisites:
  - ml-production-systems
learning_objectives:
  - Write minimal reproducible images with pinned dependencies
  - Separate build, runtime, data, and secret boundaries
  - Define health, resource, and signal behavior for services
  - Scan and test images before deployment
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Docker
supported_versions: 27.x
---

# Docker foundations

An image is an immutable filesystem plus metadata; a container is a runtime
instance with its own process, network, and resource boundaries. Build from a
small pinned base, use a lockfile and multi-stage builds, run as a non-root
user, and keep credentials out of layers and logs. A tag such as `latest` is
not a reproducible dependency.

The container contract includes the command, listening interface, health check,
graceful signal handling, writable paths, CPU/memory limits, and architecture.
Health checks must test readiness semantics, not merely that a process exists.
Build context and image provenance matter: exclude datasets and secrets, emit a
digest/SBOM, scan dependencies, and retain the exact Dockerfile and build args.

## Completion criteria

- [ ] Image dependencies and base digest are pinned.
- [ ] Runtime is non-root with bounded resources and explicit health behavior.
- [ ] Secrets and training data cannot enter image layers.
- [ ] Build, scan, run, and rollback evidence is recorded.
