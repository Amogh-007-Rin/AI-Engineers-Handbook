# Reproducibility checklist

Use this before requesting review of a lab, experiment, or project.

## Identity and environment

- [ ] Record repository URL, immutable commit, working-tree state, date, and experiment identifier.
- [ ] Record operating system, architecture, Python, dependency lock, hardware, accelerator/runtime versions, locale, and relevant environment variables.
- [ ] Keep secrets out of commands and logs; list required variable *names* only.

## Inputs and procedure

- [ ] Record dataset/model source, version, license, checksum, split manifest, schema, and preprocessing configuration.
- [ ] Record seeds and every source of randomness or nondeterminism.
- [ ] Preserve the exact command, configuration, starting checkpoint, and ordered steps from clean checkout to result.
- [ ] State CPU/RAM/storage/network/accelerator needs and measured runtime.

## Outputs and interpretation

- [ ] Save raw per-example or per-run evidence needed to recompute summaries, while respecting privacy and retention policy.
- [ ] Define metrics, thresholds, units, aggregation, confidence or variability, and acceptance ranges before interpreting results.
- [ ] Compare a credible baseline and report failed runs and negative findings.
- [ ] Distinguish measured results from assumptions, estimates, and anecdotes.
- [ ] State what the experiment does not establish.

## Independent rerun

- [ ] A second person or clean automated environment follows only the published instructions.
- [ ] Differences are quantified against a declared tolerance and explained.
- [ ] The evidence record names reviewer, role, date, platform, commands, findings, remediation, and final candidate commit.

A fixed seed and passing unit tests are necessary in some workflows but are not,
by themselves, proof of reproducibility.
