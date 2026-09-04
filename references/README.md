# AI engineering reference

This section is a concise lookup companion to the curriculum, not a replacement
for lessons or official API documentation.

## Quick entry points

- [Glossary](glossary.md): shared language for data, modeling, evaluation,
  systems, security, and research.
- [Reproducibility checklist](../shared/reproducibility-checklist.md): evidence
  to capture before calling a result repeatable.
- [Dataset card template](../templates/dataset-card-template.md) and
  [model card template](../templates/model-card-template.md).
- [Threat-model template](../templates/threat-model-template.md) and
  [experiment report](../templates/experiment-report-template.md).
- [Library primary sources](../Libraries/SOURCES.md): official documentation,
  source repositories, standards, and primary papers.
- [Security reporting](../SECURITY.md) and [maintenance policy](../MAINTENANCE.md).

For troubleshooting, first reduce the problem to the smallest deterministic
case, record versions and hardware, capture the complete error, and classify
whether the failure is data, logic, numerical, dependency, resource, network,
or permission related. Search the relevant academy's `troubleshooting/` or
lesson guidance before changing several variables at once.

## Trust hierarchy

For version-sensitive claims prefer, in order: the exact installed behavior and
tests; versioned official documentation or standards; upstream release notes
and source; primary research; then reputable secondary explanation. Record the
version and access date. A blog post or generated answer is a lead, not evidence.
