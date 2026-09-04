#!/usr/bin/env python3
"""Validate handbook front matter, links, slugs, and prerequisite graphs.

The validator deliberately uses only the Python standard library. Run it from
the repository root with: python3 scripts/validate_content.py
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOTS = (
    ROOT / "curriculum",
    ROOT / "Libraries",
    ROOT / "projects",
    ROOT / "specializations",
    ROOT / "assessments",
)
VALID_STATUS = {"outline", "draft", "review", "published", "maintenance"}
VALID_LEVEL = {"foundation", "practitioner", "advanced", "maintainer"}
VALID_COMPUTE = {"cpu", "free-gpu", "gpu-optional", "external-service"}
VALID_FORMATS = {"lesson", "notebook", "exercise", "project", "assessment", "reference"}
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LINK_RE = re.compile(r"\[[^]]*]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
PUBLIC_DOC_ROOTS = (ROOT / "templates", ROOT / "reports", ROOT / ".github")


@dataclass(frozen=True)
class Document:
    path: Path
    metadata: dict[str, object]


def scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value) if "." in value else int(value)
    return value


def parse_front_matter(path: Path) -> dict[str, object] | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("front matter is missing its closing ---") from None

    data: dict[str, object] = {}
    active_list: str | None = None
    for number, raw in enumerate(lines[1:end], start=2):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("  - "):
            if active_list is None:
                raise ValueError(f"line {number}: list item has no key")
            value = scalar(raw[4:])
            assert isinstance(data[active_list], list)
            data[active_list].append(value)
            continue
        match = re.fullmatch(r"([a-z_]+):(?:\s*(.*))?", raw)
        if not match:
            raise ValueError(f"line {number}: unsupported YAML syntax")
        key, raw_value = match.groups()
        if key in data:
            raise ValueError(f"line {number}: duplicate key {key!r}")
        if raw_value:
            data[key] = scalar(raw_value)
            active_list = None
        else:
            data[key] = []
            active_list = key
    return data


def validate_metadata(doc: Document) -> list[str]:
    data = doc.metadata
    errors: list[str] = []
    required = {
        "title", "slug", "level", "stage", "estimated_hours", "prerequisites",
        "learning_objectives", "formats", "compute", "status", "last_verified",
    }
    allowed = required | {"library", "supported_versions"}
    for key in sorted(required - data.keys()):
        errors.append(f"missing required field {key!r}")
    for key in sorted(data.keys() - allowed):
        errors.append(f"unsupported field {key!r}")

    slug = data.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        errors.append("slug must be lowercase kebab-case")
    if data.get("level") not in VALID_LEVEL:
        errors.append(f"level must be one of {sorted(VALID_LEVEL)}")
    if data.get("status") not in VALID_STATUS:
        errors.append(f"status must be one of {sorted(VALID_STATUS)}")
    if data.get("compute") not in VALID_COMPUTE:
        errors.append(f"compute must be one of {sorted(VALID_COMPUTE)}")
    hours = data.get("estimated_hours")
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0:
        errors.append("estimated_hours must be a positive number")
    for key in ("prerequisites", "learning_objectives", "formats"):
        if not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")
    objectives = data.get("learning_objectives")
    if isinstance(objectives, list) and not objectives:
        errors.append("learning_objectives must not be empty")
    formats = data.get("formats")
    if isinstance(formats, list) and any(item not in VALID_FORMATS for item in formats):
        errors.append(f"formats must contain only {sorted(VALID_FORMATS)}")
    verified = data.get("last_verified")
    try:
        if not isinstance(verified, str):
            raise ValueError
        dt.date.fromisoformat(verified)
    except ValueError:
        errors.append("last_verified must be an ISO date (YYYY-MM-DD)")
    if data.get("status") == "published" and not data.get("learning_objectives"):
        errors.append("published content requires learning objectives")
    return errors


def validate_links(path: Path, *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    targets = LINK_RE.findall(text) + HTML_LINK_RE.findall(text)
    for target in targets:
        target = target.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        decoded = target.replace("%20", " ")
        candidate = (path.parent / decoded).resolve()
        root = root.resolve()
        if root not in candidate.parents and candidate != root:
            errors.append(f"link escapes repository: {target}")
        elif not candidate.exists():
            errors.append(f"broken local link: {target}")
    return errors


def find_public_markdown() -> list[Path]:
    paths = set(ROOT.glob("*.md"))
    for root in (*CONTENT_ROOTS, *PUBLIC_DOC_ROOTS):
        if root.exists():
            paths.update(root.rglob("*.md"))
    return sorted(paths)


def find_documents() -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    errors: list[str] = []
    for root in CONTENT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            try:
                metadata = parse_front_matter(path)
            except ValueError as exc:
                errors.append(f"{path.relative_to(ROOT)}: {exc}")
                continue
            if metadata is not None:
                documents.append(Document(path, metadata))
    return documents, errors


def validate_graph(documents: list[Document]) -> list[str]:
    errors: list[str] = []
    by_slug: dict[str, Document] = {}
    for doc in documents:
        slug = doc.metadata.get("slug")
        if not isinstance(slug, str):
            continue
        if slug in by_slug:
            errors.append(
                f"duplicate slug {slug!r}: {by_slug[slug].path.relative_to(ROOT)} and "
                f"{doc.path.relative_to(ROOT)}"
            )
        by_slug[slug] = doc

    graph: dict[str, list[str]] = {}
    for slug, doc in by_slug.items():
        prerequisites = doc.metadata.get("prerequisites", [])
        graph[slug] = [item for item in prerequisites if isinstance(item, str)]
        for prerequisite in graph[slug]:
            if prerequisite not in by_slug:
                errors.append(f"{doc.path.relative_to(ROOT)}: unknown prerequisite {prerequisite!r}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(slug: str, trail: list[str]) -> None:
        if slug in visiting:
            cycle = " -> ".join(trail + [slug])
            errors.append(f"prerequisite cycle: {cycle}")
            return
        if slug in visited:
            return
        visiting.add(slug)
        for dependency in graph.get(slug, []):
            if dependency in graph:
                visit(dependency, trail + [slug])
        visiting.remove(slug)
        visited.add(slug)

    for slug in graph:
        visit(slug, [])
    return errors


def main() -> int:
    documents, errors = find_documents()
    markdown_files = find_public_markdown()
    for path in markdown_files:
        for error in validate_links(path):
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    for doc in documents:
        for error in validate_metadata(doc):
            errors.append(f"{doc.path.relative_to(ROOT)}: {error}")
    errors.extend(validate_graph(documents))
    if errors:
        print(f"Content validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(documents)} metadata document(s) and "
        f"{len(markdown_files)} public Markdown file(s); links and graph are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
