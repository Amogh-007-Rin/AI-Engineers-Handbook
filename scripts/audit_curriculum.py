#!/usr/bin/env python3
"""Audit concept-first curriculum tracks for complete learn/practice/assess evidence."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "curriculum"


def substantive(path, minimum):
    return path.is_file() and len(path.read_text(encoding="utf-8")) >= minimum


def evidence(track):
    lesson_files = [p for p in track.rglob("README.md") if p.parent != track and p.parent.name != "project"]
    project = track / "project"
    return {
        "guide": substantive(track / "README.md", 250),
        "lesson": any(substantive(p, 800) and p.read_text(encoding="utf-8").startswith("---\n") for p in lesson_files),
        "project": any(substantive(p, 100) for p in project.glob("*.py") if not p.name.startswith("test")),
        "tests": any(substantive(p, 100) for p in project.glob("test*.py")),
        "solution": any(substantive(p, 180) for p in project.glob("solution*.md")),
        "assessment": substantive(track / "assessment.md", 200),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    tracks = sorted(path for path in CURRICULUM.iterdir() if path.is_dir())
    rows = [(track.name, evidence(track)) for track in tracks]
    keys = list(rows[0][1]) if rows else []
    complete = sum(all(row.values()) for _, row in rows)
    print(f"Curriculum tracks complete: {complete}/{len(rows)}")
    print("track\t" + "\t".join(keys))
    for name, row in rows:
        print(name + "\t" + "\t".join("yes" if row[key] else "NO" for key in keys))
    if args.strict and complete != len(rows):
        print(f"Strict audit failed: {len(rows) - complete} track(s) lack required evidence.")
        return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
