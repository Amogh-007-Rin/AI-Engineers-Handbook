# Detectron2 Academy

> Status: Academy guide · Versions: verify before each release · Core compute: CPU/free tier · Last reviewed: 2026-09-03

Facebook AI Research's platform for object detection and segmentation. This academy develops transferable judgment as well as tool fluency, from first setup through advanced and production use.

## Position in the handbook

- **Track:** Computer vision
- **Prerequisites:** Python, NumPy, image fundamentals, deep learning, and vision evaluation.
- **Curriculum relationship:** Study the matching concept-first curriculum before or alongside this academy. This page owns Detectron2-specific workflows and does not replace underlying theory.

## Use it when

Use Detectron2 when its abstractions match the problem, its ecosystem satisfies operational constraints, and a measured comparison supports the choice. Record the data size, latency, correctness, maintainability, team skill, licensing, and deployment assumptions behind the decision.

## Avoid it when

Do not select Detectron2 only because it is popular or already listed here. Prefer a simpler standard-library solution, database, adjacent academy, managed service, or lower-level framework when it produces a safer and more maintainable system. The advanced assessment requires a comparison with at least one credible alternative.

## Level 0 — Setup and first success

1. Open the official installation and compatibility documentation; record the version and date used.
2. Create an isolated environment and install the smallest supported configuration.
3. Import or start Detectron2, print/report its version, and run one minimal operation.
4. Capture environment, operating system, Python/runtime, hardware, and exact command.
5. Repeat from a clean environment. A copied screenshot is not reproducibility evidence.

**Gate:** Another learner can reproduce the first success from your instructions without guessing.

## Level 1 — Foundations

- Image and annotation representations.
- Loading, transforms, inference, and visualization.
- Task metrics and dataset splits.

For each concept, predict behavior before execution, inspect types/shapes/state, add one normal assertion, and trigger one expected failure. Build a glossary mapping Detectron2 terminology to the underlying concept.

**Gate:** Complete a small task from a blank file, explain every major object, and debug deliberately broken input without copying a finished tutorial.

## Level 2 — Core workflows

- Augmentation or model-training pipelines.
- Pretrained assets and transfer learning.
- Error analysis by image and class attributes.

Build one end-to-end workflow using a small, legally reusable dataset or local fixture. Separate configuration, core logic, I/O, and presentation. Test boundaries and record expected output.

**Guided project:** Build and evaluate a small vision pipeline with visual error analysis.

**Gate:** The workflow runs twice with equivalent results, tests its key contract, and documents assumptions and known limitations.

## Level 3 — Intermediate practice

- Compose reusable components instead of growing a single notebook or script.
- Exercise configuration, serialization, integration, and extension points that practitioners use.
- Diagnose malformed input, incompatible versions, partial failure, and resource exhaustion.
- Compare at least two valid approaches with correctness checked before performance.
- Read the relevant official guide and API reference, then explain why the selected interface fits.

**Gate:** A reviewer can change configuration or input data without rewriting the system and receives actionable failures when contracts are violated.

## Level 4 — Advanced practice

- Custom models, transforms, or training hooks.
- Throughput, memory, precision, and export.
- Video, multimodal, or distributed workflows where relevant.

Read one relevant part of the project source or architecture documentation. Trace a high-level call to its underlying execution path and use that understanding to explain a measured behavior or debug a failure.

**Independent project:** Deliver an optimized vision service or training system with robustness and operational tests.

**Gate:** Defend the design against an alternative using measured evidence, known limitations, and operational constraints.

## Level 5 — Production practice

- Reproducibility and serving.
- Monitoring quality and data drift.
- Privacy, bias, adversarial, and licensing risks.

Create an operational checklist covering configuration, secrets, permissions, logs/metrics, failure recovery, dependency pinning, upgrade testing, artifact/data retention, and rollback. Include only items relevant to Detectron2, but justify omissions.

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
2. Understanding of Detectron2's mental model and appropriate API use.
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

Use the official Detectron2 documentation, release notes, source repository, and primary papers as authoritative sources. Record exact links in project work. Never infer current APIs from this guide alone: verify version-sensitive behavior and update this page's review date when evidence changes.


## Primary sources

Start with the [Primary source registry](../SOURCES.md), then cite the exact official page, version, access date, release note, or primary paper used for each version-sensitive claim.
