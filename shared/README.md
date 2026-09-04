# Shared learning infrastructure

This directory contains cross-cutting contracts that should stay consistent
across curriculum stages and library academies. Tool-specific explanations
belong in their academy; concept teaching belongs in the curriculum.

- [Reproducibility checklist](reproducibility-checklist.md)
- [Templates](../templates/README.md) for lessons, projects, experiments, cards,
  threats, and reviews
- [Core glossary](../references/glossary.md)
- [Project ladder](../projects/README.md)

Shared code added here must be dependency-light, typed where useful, documented,
covered by boundary tests, and owned by more than one consumer. Do not create a
generic utility until two real learning artifacts need the same behavior.
