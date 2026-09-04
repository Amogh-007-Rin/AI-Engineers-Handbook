#!/usr/bin/env python3
"""Run all dependency-free curriculum and academy contract suites."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRICULUM = [
    "agents", "career", "computer-vision", "data", "deep-learning", "foundations",
    "generative-ai", "graph-learning", "machine-learning", "mathematics", "ml-systems",
    "nlp-and-speech", "recommender-systems", "reinforcement-learning", "research",
    "responsible-ai", "time-series",
]

ACADEMIES = [
    "Airflow", "BentoML", "DeepFace", "Detectron2", "Docker", "Kubernetes", "MLflow",
    "MMDetection", "OpenAI SDK", "PettingZoo", "PySpark", "RLlib", "Rasa", "Ray",
    "Stable-Baselines3", "TensorFlow Serving", "TorchServe", "Ultralytics YOLO",
    "Weights & Biases",
]

ADDITIONAL_LABS = [
    ("curriculum/foundations/orientation-lab", ROOT / "curriculum" / "foundations" / "00-orientation" / "lab"),
    ("curriculum/foundations/python-lab", ROOT / "curriculum" / "foundations" / "01-python-foundations" / "lab"),
    ("curriculum/mathematics/linear-algebra-lab", ROOT / "curriculum" / "mathematics" / "01-linear-algebra" / "lab"),
]


def suites():
    for name in CURRICULUM:
        yield f"curriculum/{name}", ROOT / "curriculum" / name / "project"
    for name in ACADEMIES:
        yield f"academy/{name}", ROOT / "Libraries" / name / "projects"
    yield from ADDITIONAL_LABS


def main():
    failures = []
    for label, directory in suites():
        if not directory.is_dir():
            failures.append(f"{label}: missing {directory.relative_to(ROOT)}")
            continue
        result = subprocess.run(
            [sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", str(directory), "-v"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            failures.append(f"{label}:\n{result.stdout}")
            print(f"FAIL {label}")
        else:
            count = result.stdout.count(" ... ok")
            print(f"PASS {label} ({count} tests)")
    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        return 1
    print(f"All {len(CURRICULUM) + len(ACADEMIES) + len(ADDITIONAL_LABS)} dependency-free suites passed.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
