#!/usr/bin/env python3
"""Execute output-clean, dependency-free handbook notebooks."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: expected notebook format 4")
    namespace, count = {"__name__": "__notebook__"}, 0
    for index, cell in enumerate(notebook.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        count += 1
        if cell.get("outputs"):
            raise ValueError(f"{path}: cell {index} has committed outputs")
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
    if not count:
        raise ValueError(f"{path}: no executable code cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        raise SystemExit("No notebooks found")
    for path in paths:
        execute(path)
        print(f"PASS {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
