# Contributing

Thank you for helping build the AI Engineers Handbook. Contributions should improve a complete learner journey, not only increase the number of pages.

## Before writing

1. Read [the project blueprint](project.md).
2. Search existing curriculum and library academies for overlap.
3. Open or select an issue that states the learner problem, prerequisites, outcomes, and validation method.
4. Use the templates in `templates/`.

## Content requirements

- Write measurable learning objectives and connect every exercise to one of them.
- Explain concepts and tradeoffs; do not submit API lists or copied documentation.
- Prefer primary sources and official documentation.
- Make core examples runnable on CPU or free compute.
- Include expected output, realistic mistakes, tests, and completion criteria.
- Document versions, assets, licenses, security implications, and verification date.
- Keep solutions separate from learner-facing exercises.

## Local checks

Run these from the repository root:

```bash
python3 scripts/validate_content.py
python3 scripts/audit_lesson_contracts.py
python3 scripts/audit_library_academies.py --strict
python3 scripts/audit_curriculum.py --strict
python3 scripts/audit_learning_system.py
python3 scripts/audit_release_readiness.py
python3 scripts/execute_notebooks.py
python3 scripts/run_contract_tests.py
python3 scripts/scan_repository.py
python3 -m unittest discover -s tests
git diff --check
```

## Review gates

Published content requires technical, pedagogical, executability, and accessibility review. Security-sensitive material also requires a security and responsible-use review. Reviewers should run the artifact rather than judging prose alone.

Record reviews with [the review template](templates/review-record-template.md)
and follow [the maintenance policy](MAINTENANCE.md). A checked box or approving
comment without commit, scope, commands, findings, and reviewer role is not
release evidence.

## Pull requests

Keep each pull request focused on one complete learning slice. Describe the intended learner, outcomes, evidence, compute used, tests run, and any follow-up maintenance responsibility.
