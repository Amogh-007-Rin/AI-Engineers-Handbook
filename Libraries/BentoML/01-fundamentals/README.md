---
title: BentoML services runners models and deployment contracts
slug: bentoml-foundations
level: practitioner
stage: ml-systems
estimated_hours: 14
prerequisites:
  - ml-production-systems
learning_objectives:
  - Separate model artifacts, service APIs, runners, and deployment config
  - Define typed input/output, batching, timeout, and concurrency contracts
  - Package reproducible services without embedding secrets
  - Test health, readiness, observability, and rollback behavior
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: BentoML
supported_versions: 1.x
---

# BentoML foundations

BentoML separates a versioned model artifact from a service API and its runtime
deployment. Define input/output schemas, batch limits, timeouts, concurrency,
and error responses before decorating endpoints. Model runners may batch or
parallelize work, so preserve request order and test cancellation and back
pressure rather than assuming one request equals one model call.

Package only the dependencies and assets required for inference. Keep secrets in
the deployment environment, record model/data/code hashes, and emit structured
logs with request IDs. Health, readiness, and model-load failures need distinct
responses. Promote immutable bundles through a canary and retain a rollback
bundle with compatible schema.

## Completion criteria

- [ ] Service schemas and limits are explicit and tested.
- [ ] Model bundle, environment, and provenance are immutable.
- [ ] Readiness, errors, timeouts, and back pressure are observable.
- [ ] Canary and rollback criteria are documented.
