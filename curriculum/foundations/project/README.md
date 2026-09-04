---
title: Command-line data explorer foundation project
slug: software-foundations-project
level: foundation
stage: foundations
estimated_hours: 10
prerequisites:
  - python-foundations
learning_objectives:
  - Parse a small CSV file with explicit validation and actionable errors
  - Separate input output domain logic and command-line concerns
  - Test normal empty malformed and missing-value cases
  - Package reproducible commands and document assumptions
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Command-line data explorer

Build a dependency-free CSV profiler that reports row count, columns, missing
cells, and numeric summaries. The starter isolates pure row profiling from file
I/O. Run `python3 -m unittest -v`, then add an `argparse` command, JSON output,
exit codes, and a README example using a legally reusable local fixture.

Passing evidence includes tests for empty files, duplicate headers, inconsistent
rows, missing values, numeric conversion, and nonexistent paths; deterministic
output; a Git history with focused commits; and a short debugging retrospective.
