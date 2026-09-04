#!/usr/bin/env python3
"""Report curriculum lessons against the authored lesson contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "curriculum"


def is_lesson(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return False
    front_matter = text.split("---\n", 2)[1]
    return bool(re.search(r"^\s*- lesson\s*$", front_matter, re.MULTILINE))


def evidence(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    body = text.split("---\n", 2)[-1]
    lab_directory = path.parent / "lab"
    return {
        "depth": len(re.findall(r"\b[\w'-]+\b", body)) >= 1_000,
        "example": "worked example" in lower or ("```python" in lower and "example" in lower),
        "lab": "lab" in lower and lab_directory.is_dir()
        and any(lab_directory.glob("*.py")) and any(lab_directory.glob("test*.py")),
        "failures": any(term in lower for term in ("misconception", "failure practice", "debugging")),
        "exercises": bool(re.search(r"^##+ (?:exercises?|practice)\s*$", lower, re.MULTILINE)),
        "check": "knowledge check" in lower,
        "completion": "completion criteria" in lower,
        "summary": bool(re.search(r"^##+ summary", lower, re.MULTILINE)),
        "sources": "further reading" in lower and "https://" in lower,
        "compute": "cpu" in lower and any(term in lower for term in ("runtime", "minutes", "hours")),
        "accessibility": any(term in lower for term in ("accessibility", "screen reader", "color is", "colour is")),
        "expected": "expected" in lower and "output" in lower,
    }


def lessons(root: Path = CURRICULUM) -> list[Path]:
    return sorted(path for path in root.rglob("README.md") if is_lesson(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = [(path, evidence(path)) for path in lessons()]
    complete = sum(all(result.values()) for _, result in rows)
    print(f"Curriculum lesson contracts complete: {complete}/{len(rows)}")
    for path, result in rows:
        missing = [name for name, passed in result.items() if not passed]
        label = path.relative_to(ROOT)
        print(f"{'PASS' if not missing else 'GAP '} {label}: {', '.join(missing) if missing else 'complete'}")
    if args.strict and complete != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
