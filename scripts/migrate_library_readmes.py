#!/usr/bin/env python3
"""Replace legacy placeholder library READMEs with academy learning guides.

This is an idempotent repository-maintenance utility. NumPy, Pandas, and the
Libraries index are intentionally excluded because they have authored content.
Run from any directory with: python3 scripts/migrate_library_readmes.py
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ROOT / "Libraries"
SKIP = {"NumPy", "Pandas", "Z-Roadmap"}


@dataclass(frozen=True)
class Track:
    label: str
    prerequisites: str
    fundamentals: tuple[str, ...]
    workflows: tuple[str, ...]
    advanced: tuple[str, ...]
    production: tuple[str, ...]
    guided_project: str
    independent_project: str


TRACKS = {
    "data": Track(
        "Data and scientific computing", "Python foundations, data quality, and basic statistics",
        ("mental model and core data structures", "construction, selection, transformation, and aggregation", "types, missing values, shapes, indexes, or schemas"),
        ("ingestion-to-analysis workflow", "validation and reproducibility", "interoperability with the Python data ecosystem"),
        ("performance profiling and memory behavior", "extension and customization", "numerical or data-quality edge cases"),
        ("testing and deterministic execution", "versioned pipelines and observability", "scaling limits and engine-selection decisions"),
        "Build a reproducible exploratory analysis with executable data contracts.",
        "Build and benchmark a production-style data transformation package against a credible alternative.",
    ),
    "visualization": Track(
        "Visualization", "Python foundations, data literacy, and descriptive statistics",
        ("figure, mark, scale, axis, and layout mental models", "core chart construction", "labels, legends, themes, and accessible color"),
        ("tidy-data plotting workflows", "interactive or composable views", "export and reproducible reporting"),
        ("custom components and callbacks", "large-data rendering and profiling", "perceptual accuracy and uncertainty communication"),
        ("accessibility and deterministic exports", "dashboard testing and deployment where applicable", "security and performance boundaries"),
        "Create an accessible analysis report that matches chart choice to analytical questions.",
        "Build and evaluate a reusable visualization or dashboard with usability and performance tests.",
    ),
    "classical-ml": Track(
        "Classical machine learning", "Python, statistics, data quality, and classical ML foundations",
        ("estimator and data interface", "fit, predict, transform, and evaluate lifecycle", "task-appropriate preprocessing and metrics"),
        ("leakage-safe pipelines", "cross-validation and tuning", "error analysis and interpretability"),
        ("custom objectives or estimators", "performance and resource optimization", "uncertainty, calibration, and difficult data regimes"),
        ("reproducible training and model persistence", "serving, monitoring, and drift", "security, compatibility, and rollback"),
        "Train and defend a leakage-safe baseline and tuned model on a small public dataset.",
        "Deliver a versioned ML system with error analysis, model card, inference contract, and monitoring plan.",
    ),
    "deep-learning": Track(
        "Deep learning", "Python, linear algebra, calculus, probability, and neural-network foundations",
        ("tensor, module, parameter, and automatic-differentiation model", "input pipelines and training loops", "losses, optimizers, metrics, and checkpoints"),
        ("custom models and reusable training components", "regularization and diagnostic workflows", "pretrained models and transfer learning"),
        ("compilation, mixed precision, and distributed execution", "profiling memory and throughput", "custom operations and framework internals"),
        ("reproducibility and artifact management", "export, serving, observability, and rollback", "accelerator, security, and dependency boundaries"),
        "Train, evaluate, checkpoint, and reproduce a small neural model with documented failure analysis.",
        "Implement and benchmark an advanced model, then export or serve it with operational tests.",
    ),
    "nlp": Track(
        "NLP and generative AI", "Python, data quality, probability, NLP, and transformer foundations as applicable",
        ("text representation and library data model", "tokenization, preprocessing, inference, and evaluation", "pretrained assets and model/data licenses"),
        ("task pipelines and fine-tuning where applicable", "batching, retrieval, embeddings, or generation", "domain-specific error analysis"),
        ("custom components and adaptation", "efficiency, quantization, or distributed processing", "multilingual, robustness, and evaluation limits"),
        ("safe serving and observability", "privacy, prompt injection, and abuse considerations", "cost, latency, provenance, and version migration"),
        "Build an evaluated text pipeline with a baseline and documented failure taxonomy.",
        "Deliver a domain NLP or generative system with reproducible evaluation, safety analysis, and deployment contract.",
    ),
    "vision": Track(
        "Computer vision", "Python, NumPy, image fundamentals, deep learning, and vision evaluation",
        ("image and annotation representations", "loading, transforms, inference, and visualization", "task metrics and dataset splits"),
        ("augmentation or model-training pipelines", "pretrained assets and transfer learning", "error analysis by image and class attributes"),
        ("custom models, transforms, or training hooks", "throughput, memory, precision, and export", "video, multimodal, or distributed workflows where relevant"),
        ("reproducibility and serving", "monitoring quality and data drift", "privacy, bias, adversarial, and licensing risks"),
        "Build and evaluate a small vision pipeline with visual error analysis.",
        "Deliver an optimized vision service or training system with robustness and operational tests.",
    ),
    "rl": Track(
        "Reinforcement learning", "Python, probability, optimization, RL foundations, and a deep-learning framework",
        ("environment, observation, action, reward, and episode interfaces", "registration, wrappers, seeding, and rollout", "baseline agents and evaluation"),
        ("vectorized environments and training workflows", "logging, callbacks, checkpoints, and tuning", "multi-seed comparison and reward diagnostics"),
        ("custom environments or algorithms", "parallel or multi-agent execution", "offline, recurrent, or distributed workflows where supported"),
        ("reproducibility and experiment tracking", "policy serving and monitoring", "reward hacking, safety, and sim-to-real limits"),
        "Create a validated environment and compare a random baseline with a trained policy across seeds.",
        "Build a reproducible RL study with ablations, confidence intervals, failure analysis, and safety review.",
    ),
    "mlops": Track(
        "MLOps and AI systems", "Python, APIs, testing, containers, ML lifecycle, and basic cloud/Linux concepts",
        ("core resources, configuration, and lifecycle", "local setup and first end-to-end workflow", "artifacts, metadata, state, and permissions"),
        ("automation, composition, and integrations", "testing, logging, and failure recovery", "development-to-production workflow"),
        ("scaling, extension, and performance tuning", "high availability and distributed behavior", "upgrades, migrations, and internals"),
        ("security, secrets, and supply chain", "observability, SLOs, backup, and rollback", "capacity, cost, incident response, and disaster recovery"),
        "Build a local, tested workflow that packages or operates a small model artifact.",
        "Deliver a production-style AI service or pipeline with monitoring, threat model, load evidence, and runbook.",
    ),
    "graph": Track(
        "Graphs and graph learning", "Python, linear algebra, graph theory, data quality, and deep learning where applicable",
        ("nodes, edges, attributes, adjacency, and library data model", "construction, traversal, transformation, and visualization", "graph statistics and dataset validation"),
        ("sampling, batching, features, and task workflows", "node, edge, and graph evaluation", "interoperability and reproducibility"),
        ("large-graph performance and distributed execution", "custom algorithms or message passing", "heterogeneous, temporal, or dynamic graphs"),
        ("artifact versioning and serving", "monitoring graph and prediction drift", "privacy, fairness, and leakage across graph splits"),
        "Analyze and validate a real network, then communicate structural findings.",
        "Build an evaluated graph algorithm or GNN pipeline with leakage-safe splits and scaling evidence.",
    ),
}


CATEGORIES = {
    "data": {"SciPy", "Statsmodels", "Polars", "DuckDB", "Dask", "PySpark"},
    "visualization": {"Matplotlib", "Seaborn", "Plotly", "Bokeh", "Dash"},
    "classical-ml": {"Scikit-Learn", "XGBoost", "LightGBM", "CatBoost", "PyCaret", "Optuna", "Prophet", "Orbit", "ARIMA-SARIMA"},
    "deep-learning": {"PyTorch", "TensorFlow", "Keras", "JAX", "Flax", "FastAI", "ONNX"},
    "nlp": {"NLTK", "spaCy", "Gensim", "SentenceTransformers", "HuggingFace Transformers", "HuggingFace Diffusers", "OpenAI SDK", "Rasa"},
    "vision": {"OpenCV", "Albumentations", "Ultralytics YOLO", "MMDetection", "Detectron2", "DeepFace"},
    "rl": {"Gymnasium", "Stable-Baselines3", "RLlib", "PettingZoo"},
    "mlops": {"Fastapi", "MLflow", "Weights & Biases", "BentoML", "TensorFlow Serving", "TorchServe", "Ray", "Airflow", "Docker", "Kubernetes"},
    "graph": {"NetworkX", "PyTorch Geometric", "DGL"},
}


def description_map() -> dict[str, str]:
    """Recover descriptions from migrated pages or the legacy catalog."""
    descriptions: dict[str, str] = {}
    for folder in LIBRARIES.iterdir():
        if not folder.is_dir() or folder.name in SKIP:
            continue
        readme = folder / ("readme.md" if folder.name == "Fastapi" else "README.md")
        if not readme.exists():
            continue
        match = re.search(r"\n\n([^\n]+?) This academy develops", readme.read_text(encoding="utf-8"))
        if match:
            descriptions[folder.name] = match.group(1)
    result = subprocess.run(
        ["git", "show", "HEAD:README.md"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    pattern = re.compile(r"\|\s*\d+\s*\|\s*\[([^]]+)]\([^)]+\)\s*\|\s*(.*?)\s*\|")
    for name, description in pattern.findall(result.stdout):
        descriptions.setdefault(name, description)
    descriptions.setdefault("Fastapi", "A modern Python framework for building typed, high-performance APIs.")
    return descriptions


def category_for(name: str) -> str:
    matches = [category for category, names in CATEGORIES.items() if name in names]
    if len(matches) != 1:
        raise ValueError(f"{name!r} must belong to exactly one category; got {matches}")
    return matches[0]


def bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item.capitalize()}." for item in items)


def render(name: str, description: str, track: Track) -> str:
    display = "FastAPI" if name == "Fastapi" else name
    return f"""# {display} Academy

> Status: Academy guide · Versions: verify before each release · Core compute: CPU/free tier · Last reviewed: 2026-09-03

{description} This academy develops transferable judgment as well as tool fluency, from first setup through advanced and production use.

## Position in the handbook

- **Track:** {track.label}
- **Prerequisites:** {track.prerequisites}.
- **Curriculum relationship:** Study the matching concept-first curriculum before or alongside this academy. This page owns {display}-specific workflows and does not replace underlying theory.

## Use it when

Use {display} when its abstractions match the problem, its ecosystem satisfies operational constraints, and a measured comparison supports the choice. Record the data size, latency, correctness, maintainability, team skill, licensing, and deployment assumptions behind the decision.

## Avoid it when

Do not select {display} only because it is popular or already listed here. Prefer a simpler standard-library solution, database, adjacent academy, managed service, or lower-level framework when it produces a safer and more maintainable system. The advanced assessment requires a comparison with at least one credible alternative.

## Level 0 — Setup and first success

1. Open the official installation and compatibility documentation; record the version and date used.
2. Create an isolated environment and install the smallest supported configuration.
3. Import or start {display}, print/report its version, and run one minimal operation.
4. Capture environment, operating system, Python/runtime, hardware, and exact command.
5. Repeat from a clean environment. A copied screenshot is not reproducibility evidence.

**Gate:** Another learner can reproduce the first success from your instructions without guessing.

## Level 1 — Foundations

{bullets(track.fundamentals)}

For each concept, predict behavior before execution, inspect types/shapes/state, add one normal assertion, and trigger one expected failure. Build a glossary mapping {display} terminology to the underlying concept.

**Gate:** Complete a small task from a blank file, explain every major object, and debug deliberately broken input without copying a finished tutorial.

## Level 2 — Core workflows

{bullets(track.workflows)}

Build one end-to-end workflow using a small, legally reusable dataset or local fixture. Separate configuration, core logic, I/O, and presentation. Test boundaries and record expected output.

**Guided project:** {track.guided_project}

**Gate:** The workflow runs twice with equivalent results, tests its key contract, and documents assumptions and known limitations.

## Level 3 — Intermediate practice

- Compose reusable components instead of growing a single notebook or script.
- Exercise configuration, serialization, integration, and extension points that practitioners use.
- Diagnose malformed input, incompatible versions, partial failure, and resource exhaustion.
- Compare at least two valid approaches with correctness checked before performance.
- Read the relevant official guide and API reference, then explain why the selected interface fits.

**Gate:** A reviewer can change configuration or input data without rewriting the system and receives actionable failures when contracts are violated.

## Level 4 — Advanced practice

{bullets(track.advanced)}

Read one relevant part of the project source or architecture documentation. Trace a high-level call to its underlying execution path and use that understanding to explain a measured behavior or debug a failure.

**Independent project:** {track.independent_project}

**Gate:** Defend the design against an alternative using measured evidence, known limitations, and operational constraints.

## Level 5 — Production practice

{bullets(track.production)}

Create an operational checklist covering configuration, secrets, permissions, logs/metrics, failure recovery, dependency pinning, upgrade testing, artifact/data retention, and rollback. Include only items relevant to {display}, but justify omissions.

**Gate:** Run a failure drill and demonstrate detection, diagnosis, recovery, and a prevention-oriented retrospective.

## Cookbook exercises

- Create the smallest correct example and explain every line.
- Convert a realistic raw input into the library’s preferred representation.
- Serialize or export an artifact and verify a round trip when supported.
- Add structured logging and measure one meaningful resource or quality metric.
- Reproduce a common error, reduce it to a minimal case, and document the fix.
- Compare the canonical workflow with one credible alternative on the same task.

## Final practical assessment

Submit the independent project plus a 1,000–1,500 word engineering report. Score 20 points each for:

1. Correctness and explicit edge-case handling.
2. Understanding of {display}'s mental model and appropriate API use.
3. Tests, reproducibility, versioning, and artifact/data provenance.
4. Evaluation, performance evidence, debugging, and failure recovery.
5. Tradeoff reasoning, security/responsible-use review, and communication.

A score of 80/100 is required, with no critical correctness, security, silent-data-loss, or reproducibility failure. Revisions are allowed after documenting the original failure and correction.

## Mastery checklist

- [ ] Foundation: install, explain, execute, and debug essential operations.
- [ ] Practitioner: build and test an end-to-end workflow.
- [ ] Advanced: profile, extend, integrate, and compare alternatives.
- [ ] Production: monitor, secure, upgrade, recover, and communicate limitations.
- [ ] Maintainer-ready (optional): navigate source, reproduce an issue, and prepare an upstream-quality contribution.

## Source and maintenance policy

Use the official {display} documentation, release notes, source repository, and primary papers as authoritative sources. Record exact links in project work. Never infer current APIs from this guide alone: verify version-sensitive behavior and update this page's review date when evidence changes.
"""


def render_catalog() -> str:
    lines = [
        "# Library Academy Catalog",
        "",
        "This catalog tracks all 60 library/tool/method academies. An academy guide defines the complete learning and assessment route; `Pilot content` additionally includes authored lessons and projects.",
        "",
        "| Track | Academy | Current content |",
        "|---|---|---|",
        "| Data and scientific computing | [NumPy](NumPy/README.md) | Pilot content |",
        "| Data and scientific computing | [Pandas](Pandas/README.md) | Pilot content |",
    ]
    for category, track in TRACKS.items():
        for name in sorted(CATEGORIES[category]):
            display = "FastAPI" if name == "Fastapi" else name
            readme = "readme.md" if name == "Fastapi" else "README.md"
            lines.append(f"| {track.label} | [{display}]({name}/{readme}) | Academy guide |")
    lines.extend([
        "",
        "## Status meanings",
        "",
        "- **Academy guide:** A tailored basic-to-advanced route, projects, assessment, and production checklist are available in the README.",
        "- **Pilot content:** The guide also has authored lesson and project files validated by the curriculum graph.",
        "- **Complete academy:** Reserved for academies meeting every definition-of-done requirement in `project.md`, including learner testing and domain review.",
        "",
        "`Z-Roadmap` is excluded because it is roadmap migration material, not a library or method academy.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    descriptions = description_map()
    folders = sorted(path for path in LIBRARIES.iterdir() if path.is_dir() and path.name not in SKIP)
    expected = set().union(*CATEGORIES.values())
    actual = {path.name for path in folders}
    if actual != expected:
        raise SystemExit(f"category inventory mismatch: missing={sorted(actual - expected)}, stale={sorted(expected - actual)}")
    for folder in folders:
        category = category_for(folder.name)
        description = descriptions.get(folder.name, f"{folder.name} is part of the AI engineering ecosystem.")
        readme = folder / ("readme.md" if folder.name == "Fastapi" else "README.md")
        readme.write_text(render(folder.name, description, TRACKS[category]), encoding="utf-8")
    (LIBRARIES / "CATALOG.md").write_text(render_catalog(), encoding="utf-8")
    print(f"Migrated {len(folders)} academy guides.")


if __name__ == "__main__":
    main()
