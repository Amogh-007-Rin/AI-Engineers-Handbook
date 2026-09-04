#!/usr/bin/env python3
"""Validate structured external evidence required for a stable release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports" / "release-evidence.json"
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
RUN_URL_RE = re.compile(r"^https://github\.com/[^/]+/[^/]+/actions/runs/\d+(?:/.*)?$")
REQUIRED_WORKFLOWS = {
    "quality.yml", "extended-academies.yml", "nlp-academies.yml",
    "heavy-academies.yml", "compatibility-academies.yml",
    "domain-framework-academies.yml", "mlops-academies.yml",
}
REQUIRED_GLOBAL_REVIEWS = {
    "executability", "accessibility", "security", "responsible_ai", "licensing",
}


def academy_names(root: Path = ROOT) -> list[str]:
    return sorted(
        path.name for path in (root / "Libraries").iterdir()
        if path.is_dir() and path.name != "Z-Roadmap"
        and any(child.name.casefold() == "readme.md" for child in path.iterdir())
    )


def _valid_commit(value: object) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value))


def audit(manifest: dict, *, academies: list[str], workflows: set[str]) -> list[str]:
    pending: list[str] = []
    if manifest.get("schema_version") != 1:
        pending.append("schema_version must equal 1")
    if not _valid_commit(manifest.get("commit")):
        pending.append("release candidate commit is missing or invalid")
    candidate = manifest.get("release_candidate")
    if not isinstance(candidate, str) or not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", candidate):
        pending.append("release_candidate must be a semantic version such as v1.0.0")

    successful_workflows: set[str] = set()
    for entry in manifest.get("workflow_runs", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("workflow")
        if (name in workflows and entry.get("conclusion") == "success"
                and _valid_commit(entry.get("commit"))
                and isinstance(entry.get("url"), str)
                and RUN_URL_RE.fullmatch(entry["url"])):
            successful_workflows.add(name)
    for name in sorted(workflows - successful_workflows):
        pending.append(f"successful remote run missing: {name}")

    review_coverage = {kind: set() for kind in {"technical", "pedagogy"}}
    global_coverage = {kind: set() for kind in REQUIRED_GLOBAL_REVIEWS}
    for entry in manifest.get("reviews", []):
        if not isinstance(entry, dict) or entry.get("decision") != "approved":
            continue
        if entry.get("independent") is not True or not _valid_commit(entry.get("commit")):
            continue
        scope = entry.get("academies")
        if scope == "all":
            covered = set(academies)
        elif isinstance(scope, list):
            covered = {name for name in scope if name in academies}
        else:
            covered = set()
        kind = entry.get("review_type")
        if kind in review_coverage:
            review_coverage[kind].update(covered)
        if kind in global_coverage:
            global_coverage[kind].update(covered)
    expected = set(academies)
    for kind, covered in sorted(review_coverage.items()):
        missing = expected - covered
        if missing:
            pending.append(f"{kind} review missing for {len(missing)}/{len(expected)} academies")
    for kind, covered in sorted(global_coverage.items()):
        missing = expected - covered
        if missing:
            pending.append(f"{kind} review missing for {len(missing)}/{len(expected)} academies")

    journeys: set[tuple[str, str]] = set()
    for entry in manifest.get("learner_journeys", []):
        if not isinstance(entry, dict):
            continue
        key = (entry.get("academy"), entry.get("level"))
        if (key[0] in expected and key[1] in {"foundation", "practitioner"}
                and entry.get("result") == "passed"
                and entry.get("blocking_feedback_resolved") is True
                and _valid_commit(entry.get("commit"))
                and bool(entry.get("evidence"))):
            journeys.add(key)
    for level in ("foundation", "practitioner"):
        missing = [name for name in academies if (name, level) not in journeys]
        if missing:
            pending.append(f"{level} learner journey missing for {len(missing)}/{len(expected)} academies")
    return pending


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("release evidence root must be an object")
    for key in ("workflow_runs", "reviews", "learner_journeys"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{key} must be an array")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strict", action="store_true",
                        help="fail while any stable-release evidence is pending")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Release evidence manifest invalid: {exc}")
        return 1
    academies = academy_names()
    pending = audit(manifest, academies=academies, workflows=REQUIRED_WORKFLOWS)
    if pending:
        print(f"Release evidence pending: {len(pending)} gate(s)")
        for item in pending:
            print(f"- {item}")
        return 1 if args.strict else 0
    print(f"Stable release evidence complete for {len(academies)} academies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
