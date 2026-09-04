# Airflow Academy

> Status: Academy guide · Versions: verify before each release · Core compute: CPU/free tier · Last reviewed: 2026-09-03

Platform to programmatically author, schedule, and monitor data workflows (DAGs). This academy develops transferable judgment as well as tool fluency, from first setup through advanced and production use.

## Position in the handbook

- **Track:** MLOps and AI systems
- **Prerequisites:** Python, APIs, testing, containers, ML lifecycle, and basic cloud/Linux concepts.
- **Curriculum relationship:** Study the matching concept-first curriculum before or alongside this academy. This page owns Airflow-specific workflows and does not replace underlying theory.

## Use it when

Use Airflow when its abstractions match the problem, its ecosystem satisfies operational constraints, and a measured comparison supports the choice. Record the data size, latency, correctness, maintainability, team skill, licensing, and deployment assumptions behind the decision.

## Avoid it when

Do not select Airflow only because it is popular or already listed here. Prefer a simpler standard-library solution, database, adjacent academy, managed service, or lower-level framework when it produces a safer and more maintainable system. The advanced assessment requires a comparison with at least one credible alternative.

## Level 0 — Setup and first success

1. Open the official installation and compatibility documentation; record the version and date used.
2. Create an isolated environment and install the smallest supported configuration.
3. Import or start Airflow, print/report its version, and run one minimal operation.
4. Capture environment, operating system, Python/runtime, hardware, and exact command.
5. Repeat from a clean environment. A copied screenshot is not reproducibility evidence.

**Gate:** Another learner can reproduce the first success from your instructions without guessing.

## Level 1 — Foundations

- Core resources, configuration, and lifecycle.
- Local setup and first end-to-end workflow.
- Artifacts, metadata, state, and permissions.

For each concept, predict behavior before execution, inspect types/shapes/state, add one normal assertion, and trigger one expected failure. Build a glossary mapping Airflow terminology to the underlying concept.

**Gate:** Complete a small task from a blank file, explain every major object, and debug deliberately broken input without copying a finished tutorial.

## Level 2 — Core workflows

- Automation, composition, and integrations.
- Testing, logging, and failure recovery.
- Development-to-production workflow.

Build one end-to-end workflow using a small, legally reusable dataset or local fixture. Separate configuration, core logic, I/O, and presentation. Test boundaries and record expected output.

**Guided project:** Build a local, tested workflow that packages or operates a small model artifact.

**Gate:** The workflow runs twice with equivalent results, tests its key contract, and documents assumptions and known limitations.

## Level 3 — Intermediate practice

- Compose reusable components instead of growing a single notebook or script.
- Exercise configuration, serialization, integration, and extension points that practitioners use.
- Diagnose malformed input, incompatible versions, partial failure, and resource exhaustion.
- Compare at least two valid approaches with correctness checked before performance.
- Read the relevant official guide and API reference, then explain why the selected interface fits.

**Gate:** A reviewer can change configuration or input data without rewriting the system and receives actionable failures when contracts are violated.

## Level 4 — Advanced practice

- Scaling, extension, and performance tuning.
- High availability and distributed behavior.
- Upgrades, migrations, and internals.

Read one relevant part of the project source or architecture documentation. Trace a high-level call to its underlying execution path and use that understanding to explain a measured behavior or debug a failure.

**Independent project:** Deliver a production-style AI service or pipeline with monitoring, threat model, load evidence, and runbook.

**Gate:** Defend the design against an alternative using measured evidence, known limitations, and operational constraints.

## Level 5 — Production practice

- Security, secrets, and supply chain.
- Observability, slos, backup, and rollback.
- Capacity, cost, incident response, and disaster recovery.

Create an operational checklist covering configuration, secrets, permissions, logs/metrics, failure recovery, dependency pinning, upgrade testing, artifact/data retention, and rollback. Include only items relevant to Airflow, but justify omissions.

**Gate:** Run a failure drill and demonstrate detection, diagnosis, recovery, and a prevention-oriented retrospective.

## Cookbook exercises

- Create the smallest correct example and explain every line.
- Convert a realistic raw input into the library’s preferred representation.
- Serialize or export an artifact and verify a round trip when supported.
- Add structured logging and measure one meaningful resource or quality metric.
- Reproduce a common error, reduce it to a minimal case, and document the fix.
- Compare the canonical workflow with one credible alternative on the same task.

## Final practical assessment

Submit the independent project plus a 1,000–1,500 word engineering report. Score 20 points each for:

1. Correctness and explicit edge-case handling.
2. Understanding of Airflow's mental model and appropriate API use.
3. Tests, reproducibility, versioning, and artifact/data provenance.
4. Evaluation, performance evidence, debugging, and failure recovery.
5. Tradeoff reasoning, security/responsible-use review, and communication.

A score of 80/100 is required, with no critical correctness, security, silent-data-loss, or reproducibility failure. Revisions are allowed after documenting the original failure and correction.

## Mastery checklist

- [ ] Foundation: install, explain, execute, and debug essential operations.
- [ ] Practitioner: build and test an end-to-end workflow.
- [ ] Advanced: profile, extend, integrate, and compare alternatives.
- [ ] Production: monitor, secure, upgrade, recover, and communicate limitations.
- [ ] Maintainer-ready (optional): navigate source, reproduce an issue, and prepare an upstream-quality contribution.

## Source and maintenance policy

Use the official Airflow documentation, release notes, source repository, and primary papers as authoritative sources. Record exact links in project work. Never infer current APIs from this guide alone: verify version-sensitive behavior and update this page's review date when evidence changes.
