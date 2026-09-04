#!/usr/bin/env python3
"""Report completion evidence for every library; use --strict as release gate."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ROOT / "Libraries"


def has_glob(root, pattern):
    return any(path.is_file() for path in root.rglob(pattern))


def has_substantive(root, pattern, minimum):
    return any(path.is_file() and len(path.read_text(encoding="utf-8")) >= minimum
               for path in root.rglob(pattern))


def evidence(academy):
    lessons = list(academy.rglob("[0-9][0-9]-*/README.md"))
    if not lessons:
        lessons = [p for p in academy.rglob("README.md") if p.name == "README.md" and p != academy / "README.md"]
    python_files = [p for p in academy.rglob("*.py") if "__pycache__" not in p.parts]
    project_files = [p for p in python_files if not p.name.startswith("test")]
    test_files = [p for p in python_files if p.name.startswith("test")]
    return {
        "guide": any((academy / name).is_file() and len((academy / name).read_text(encoding="utf-8")) >= 1000
                     for name in ("README.md", "readme.md")),
        "lesson": bool(lessons) and any(p.read_text(encoding="utf-8").startswith("---\n")
                                        and len(p.read_text(encoding="utf-8")) >= 800 for p in lessons),
        "exercises": has_substantive(academy, "exercises/*.md", 150),
        "project": any(len(p.read_text(encoding="utf-8")) >= 100 for p in project_files),
        "tests": any(len(p.read_text(encoding="utf-8")) >= 100 for p in test_files),
        "solution": has_substantive(academy, "solution*.md", 180),
        "assessment": has_substantive(academy, "assessment.md", 120),
        "environment": has_glob(academy, "environment/*"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    academies = sorted(p for p in LIBRARIES.iterdir() if p.is_dir() and p.name != "Z-Roadmap")
    rows = [(path.name, evidence(path)) for path in academies]
    keys = list(rows[0][1]) if rows else []
    complete = sum(all(row.values()) for _, row in rows)
    print(f"Library academies complete: {complete}/{len(rows)}")
    print("academy\t" + "\t".join(keys))
    for name, row in rows:
        print(name + "\t" + "\t".join("yes" if row[key] else "NO" for key in keys))
    if args.strict and complete != len(rows):
        print(f"Strict audit failed: {len(rows) - complete} academies lack required evidence.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
