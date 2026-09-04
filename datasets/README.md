# Dataset practice

The repository does not commit large, private, restricted, or license-unclear
datasets. Curriculum projects should use small generated fixtures for tests and
provide a reproducible acquisition path for larger optional data.

## Before using a dataset

1. Identify its owner, canonical source, exact version, retrieval date, license,
   terms, checksum, and required attribution.
2. Describe collection, annotation, unit of observation, time span, geography,
   populations, schema, missing-value meanings, known errors, and duplicates.
3. State intended uses and exclusions. Consent for collection does not imply
   suitability for every model or deployment.
4. Define splits from the deployment decision: time-, entity-, group-, or
   geography-aware splits often matter more than random rows.
5. Inspect target leakage, proxies, sensitive attributes, representation,
   label quality, privacy, malicious content, and licensing of derived assets.
6. Create a minimal deterministic fixture that tests pipeline behavior without
   redistributing the full dataset.
7. Complete a [dataset card](../templates/dataset-card-template.md) and record
   every transformation in lineage evidence.

## Download-helper contract

A helper must require an explicit destination, use HTTPS where available,
verify a published or maintainer-recorded checksum, reject unexpected archive
paths, avoid executing downloaded content, and be safe to rerun. It must explain
network, disk, RAM, runtime, deletion, and license requirements before download.
Tests use a local fixture or mocked transport and never depend on the public
network.

## Synthetic fixtures

Synthetic data can verify schemas, boundaries, failure handling, and deterministic
metrics. It cannot establish real-world validity, fairness, privacy safety, or
production performance. Label synthetic results clearly and replace them with
authorized representative evidence before making deployment claims.
