# AI Engineers Handbook

A free, open-source path from first principles to production AI engineering.

This repository is becoming a structured curriculum with runnable labs, mastery checks, projects, specialization tracks, and basic-to-advanced academies for individual tools. Read the [project blueprint](project.md) for the complete scope and standards.

## Start here

1. Read [How to learn with this handbook](curriculum/foundations/00-orientation/README.md).
2. Follow the [curriculum map](curriculum/README.md).
3. Use [Library Academies](Libraries/README.md) for tool-specific depth.
4. Complete exercises and stage gates; progress is based on evidence, not pages viewed.
5. Run the [dependency-free notebooks](notebooks/) for short executable checks.

The core path assumes basic computer literacy but no programming or advanced mathematics. Core exercises target a normal CPU or free hosted GPU.

## Learning tracks

- **Flagship:** Foundations → mathematics and data → ML → deep learning → domain AI → GenAI and agents → production and research.
- **Accelerated:** Planned placement routes for experienced developers, analysts, and mathematicians.
- **Specializations:** Advanced ML, research, NLP/LLMs, vision, GenAI/agents, RL, MLOps, data-centric AI, and responsible AI.

## Current status

All 60 library academies satisfy the substantive artifact gate, and all 17
concept-first tracks satisfy the learn-project-test-solution-assessment gate.
The resource spans foundations through domain AI, generative systems, production,
research, and career/capstone practice. Most pages remain `draft` until technical,
pedagogical, accessibility, and clean-environment reviews are recorded. See the
[verification status](reports/verification-status.md); structural completeness
must not be confused with a stable reviewed release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run:

```bash
python3 scripts/validate_content.py
python3 scripts/audit_library_academies.py --strict
python3 scripts/audit_curriculum.py --strict
python3 scripts/execute_notebooks.py
python3 scripts/run_contract_tests.py
python3 scripts/scan_repository.py
python3 -m unittest discover -s tests
git diff --check
```

## License

The repository retains its existing MIT license status. Contributions must use compatible sources and assets.
