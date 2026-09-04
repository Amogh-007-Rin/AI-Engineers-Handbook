---
title: MLflow runs artifacts registry promotion and reproducibility
slug: mlflow-foundations
level: practitioner
stage: ml-systems
estimated_hours: 14
prerequisites:
  - ml-production-systems
learning_objectives:
  - Define run, parameter, metric, artifact, and model-version contracts
  - Separate immutable experiment evidence from mutable registry aliases
  - Promote models using quality, security, and compatibility gates
  - Reproduce a run from its environment and data references
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: MLflow
supported_versions: 2.x
---

# MLflow foundations

An MLflow run is an evidence record, not a notebook scrapbook. Log the code
revision, environment lock, data/version hash, parameters, metrics with step
and split, and artifacts with schema. Metrics without their denominator,
dataset, or evaluation code are not reproducible evidence.

The model registry provides lineage, versions, aliases, and deployment
transitions. Treat a version as immutable; use aliases such as `candidate` and
`champion` only with an approval and rollback policy. Validate signature,
dependency compatibility, input limits, security scan, quality thresholds, and
monitoring before promotion. Never put secrets or raw personal data in logs.

## Completion criteria

- [ ] Run manifests contain code, data, environment, and metric provenance.
- [ ] Artifacts have schemas, hashes, and bounded content.
- [ ] Promotion has explicit quality/security/compatibility gates.
- [ ] Alias rollback and retention policies are tested.
