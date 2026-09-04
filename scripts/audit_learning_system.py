#!/usr/bin/env python3
"""Audit cross-cutting learner surfaces promised by the project blueprint."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "project ladder": ("projects/README.md", 2_000),
    "assessment index": ("assessments/README.md", 2_000),
    "graduation rubric": ("assessments/graduation-rubric.md", 1_500),
    "reference index": ("references/README.md", 800),
    "glossary": ("references/glossary.md", 2_000),
    "dataset practice": ("datasets/README.md", 1_200),
    "shared index": ("shared/README.md", 500),
    "reproducibility checklist": ("shared/reproducibility-checklist.md", 1_500),
    "contributor handbook": ("contributing/README.md", 1_200),
    "template index": ("templates/README.md", 500),
}


def inspect(root: Path = ROOT) -> dict[str, bool]:
    results = {}
    for label, (relative, minimum) in REQUIRED.items():
        path = root / relative
        results[label] = path.is_file() and len(path.read_text(encoding="utf-8")) >= minimum

    project_index = root / "projects/README.md"
    project_text = project_index.read_text(encoding="utf-8") if project_index.is_file() else ""
    numbered_steps = {int(value) for value in re.findall(r"^\|\s*(\d+)\s*\|", project_text, re.MULTILINE)}
    results["fourteen project steps"] = numbered_steps == set(range(1, 15))

    assessment_index = root / "assessments/README.md"
    assessment_text = assessment_index.read_text(encoding="utf-8") if assessment_index.is_file() else ""
    results["seventeen stage gates"] = assessment_text.count("/assessment.md)") == 17
    return results


def main() -> int:
    results = inspect()
    print("Blueprint learning surfaces")
    for label, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'} {label}")
    failures = [label for label, passed in results.items() if not passed]
    if failures:
        print(f"Learning-system audit failed: {len(failures)} requirement(s) missing.")
        return 1
    print(f"All {len(results)} cross-cutting requirements passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
