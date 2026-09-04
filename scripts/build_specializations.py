#!/usr/bin/env python3
"""Build the canonical specialization path documents from reviewed specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Track:
    title: str
    duration: str
    prerequisite: str
    outcomes: tuple[str, ...]
    modules: tuple[str, ...]
    project: str
    extension: str
    evidence: tuple[str, ...]


TRACKS = {
    "ml-engineer": Track(
        "Machine Learning Engineer", "12–16 weeks", "production-ai-project",
        ("Design leakage-safe ML products from ambiguous requirements", "Build reliable feature training evaluation and inference pipelines", "Operate models under latency cost drift and governance constraints"),
        ("Advanced tabular learning, calibration, ranking, recommendation, and forecasting", "Feature platforms, batch/online consistency, registries, and continuous training", "Testing, canaries, monitoring, retraining policy, and ML system design"),
        "Deliver a versioned decision system with offline/online feature parity, three baselines, calibrated decisions, monitored serving, and rollback.",
        "Reproduce and extend a recent data-centric, ranking, or efficient-tabular result under a declared compute budget.",
        ("Architecture and feature contracts", "Experiment lineage and model card", "Load/drift tests and incident drill"),
    ),
    "deep-learning-engineer": Track(
        "Deep Learning Engineer", "14–18 weeks", "deep-learning-framework-parity",
        ("Implement and diagnose modern neural architectures", "Optimize training and inference across hardware constraints", "Translate models among research training and serving environments"),
        ("Optimization, initialization, regularization, scaling, and distributed training", "CNN, transformer, graph, and generative architecture internals", "Compilation, mixed precision, profiling, compression, export, and serving"),
        "Train one architecture in all three core frameworks, prove parity on fixtures, profile bottlenecks, and deploy an optimized version.",
        "Reproduce an architecture or optimization paper and test one mechanism-focused ablation.",
        ("Gradient and parity tests", "Training profiles and ablations", "Export/serving compatibility report"),
    ),
    "research-engineer": Track(
        "AI Research Engineer", "16–20 weeks", "paper-reproduction-project",
        ("Turn literature gaps into falsifiable experiments", "Build reliable research infrastructure and reproduce results", "Communicate uncertainty negative findings and validity limits"),
        ("Literature synthesis, theory-to-code translation, and benchmark forensics", "Experiment configuration, artifact lineage, multi-seed statistics, and ablations", "Efficient prototyping, distributed studies, open-source artifacts, and technical writing"),
        "Reproduce a bounded paper, audit its central evidence, and release a reusable implementation with one original extension.",
        "Write a workshop-style report comparing two explanations through controlled experiments.",
        ("Preregistered protocol", "Reproduction package and raw results", "Paper, review response, and limitations"),
    ),
    "nlp-llm-engineer": Track(
        "NLP, Speech, and LLM Engineer", "14–18 weeks", "grounded-rag-project",
        ("Build multilingual text speech retrieval and generation systems", "Evaluate component and end-to-end language behavior", "Operate language systems with privacy safety latency and cost controls"),
        ("Tokenization, corpus design, embeddings, transformers, speech, and multilingual NLP", "Retrieval, reranking, RAG, adaptation, structured generation, and human evaluation", "Inference optimization, observability, provenance, prompt injection, and abuse controls"),
        "Deliver a multilingual grounded assistant with relevance judgments, citation checks, speech or text interface, red-team suite, and monitored service.",
        "Compare retrieval/adaptation strategies or reproduce an efficient language-model result.",
        ("Corpus/dataset cards", "Retrieval and generation evaluation", "Safety, latency, cost, and deployment report"),
    ),
    "vision-engineer": Track(
        "Computer Vision and Multimodal Engineer", "14–18 weeks", "vision-evaluation-project",
        ("Design vision datasets tasks models and evaluation", "Build robust image video and vision-language systems", "Optimize and deploy vision inference under device constraints"),
        ("Imaging, annotation, augmentation, detection, segmentation, tracking, and video", "CNNs, vision transformers, self-supervision, diffusion, and vision-language models", "Robustness, edge optimization, export, acceleration, monitoring, and privacy"),
        "Deliver a detection or segmentation system with annotation audit, group-safe splits, robustness slices, optimized deployment, and visual error analysis.",
        "Reproduce a vision architecture or robustness result and test it under a shifted dataset.",
        ("Annotation and dataset audit", "Model/robustness evaluation", "Deployment benchmark and risk review"),
    ),
    "genai-agent-engineer": Track(
        "Generative AI and Agent Engineer", "12–16 weeks", "safe-agent-project",
        ("Design grounded generative systems and bounded agents", "Evaluate quality safety permissions recovery cost and latency", "Operate tool-using systems with least privilege and human oversight"),
        ("RAG, adaptation, multimodality, structured outputs, and model routing", "Workflow/agent design, state, memory, tools, protocols, and approval", "Offline/online evaluation, injection defense, tracing, recovery, and economics"),
        "Deliver a capability-gated agent over a permissioned corpus with deterministic workflows, adversarial evaluation, approvals, rollback, and cost limits.",
        "Compare an agent with a deterministic workflow or reproduce an agent-evaluation method.",
        ("Threat model and capability map", "Task/policy/recovery evaluation", "Traces, runbook, and cost model"),
    ),
    "rl-engineer": Track(
        "Reinforcement Learning Engineer", "14–18 weeks", "rl-bandit-project",
        ("Implement and evaluate sequential decision algorithms", "Design environments rewards and offline/online experiments", "Diagnose instability safety and sim-to-real limitations"),
        ("Dynamic programming, value methods, policy gradients, actor-critic, and model-based RL", "Offline RL, imitation, multi-agent systems, exploration, and evaluation", "Distributed training, environment validation, policy serving, and safety"),
        "Build a custom validated environment and compare heuristic, value, and policy baselines across seeds with safety constraints.",
        "Reproduce an offline, model-based, or multi-agent result and stress-test reward assumptions.",
        ("Environment and reward specification", "Multi-seed learning evidence", "Safety and deployment boundary report"),
    ),
    "mlops-platform-engineer": Track(
        "MLOps and AI Platform Engineer", "14–18 weeks", "production-ai-project",
        ("Build secure self-service ML platform capabilities", "Operate training serving data and artifact infrastructure", "Balance reliability developer experience governance and cost"),
        ("Containers, orchestration, pipelines, registries, feature/data systems, and CI/CD/CT", "Distributed training/inference, GPU scheduling, tenancy, secrets, and supply chain", "SLOs, capacity, observability, incidents, disaster recovery, and FinOps"),
        "Deliver a local or cloud-neutral platform slice that trains, registers, deploys, monitors, rolls back, and audits a model.",
        "Benchmark two serving or orchestration designs under failure and cost constraints.",
        ("Platform API and threat model", "Load/failure/upgrade evidence", "SLO dashboard, runbooks, and cost report"),
    ),
    "data-centric-ai-engineer": Track(
        "Data-Centric AI Engineer", "12–16 weeks", "classical-ml-stage-project",
        ("Diagnose model failures through data quality and coverage", "Design annotation validation and active-learning systems", "Version and govern datasets across the ML lifecycle"),
        ("Data contracts, profiling, labeling, weak supervision, augmentation, and synthetic data", "Error taxonomies, slice discovery, active learning, deduplication, and leakage", "Dataset versioning, lineage, privacy, monitoring, and feedback loops"),
        "Improve a fixed baseline primarily through governed data interventions and prove gains with controlled experiments.",
        "Reproduce a data-selection, cleaning, labeling, or synthetic-data method on a second domain.",
        ("Dataset lineage and cards", "Intervention ablations", "Quality monitoring and annotation handbook"),
    ),
    "responsible-ai-engineer": Track(
        "Responsible AI, Security, and Evaluation Engineer", "12–16 weeks", "responsible-ai-project",
        ("Translate harms and threats into measurable evaluation", "Design governance controls and human oversight", "Red-team and monitor models data RAG and agents"),
        ("Fairness, privacy, accessibility, explainability, governance, and impact assessment", "Adversarial ML, GenAI/agent security, provenance, abuse, and incident response", "Evaluation design, human studies, red teaming, audit evidence, and regulatory mapping"),
        "Audit and harden a production-like AI system, implementing tested controls and an accountable residual-risk decision.",
        "Reproduce an evaluation or attack paper and test whether its conclusions transfer to another model/domain.",
        ("Impact and threat assessments", "Red-team dataset and control tests", "Governance record, incident exercise, and audit report"),
    ),
}


def render(slug: str, track: Track) -> str:
    objectives = "\n".join(f"  - {item}" for item in track.outcomes)
    modules = "\n".join(f"{i}. {item}." for i, item in enumerate(track.modules, 1))
    evidence = "\n".join(f"- {item}." for item in track.evidence)
    return f"""---
title: {track.title} specialization
slug: specialization-{slug}
level: advanced
stage: specialization
estimated_hours: 160
prerequisites:
  - {track.prerequisite}
learning_objectives:
{objectives}
formats:
  - lesson
  - project
  - assessment
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# {track.title} specialization

**Expected duration:** {track.duration} after the common core. Entry requires evidence for `{track.prerequisite}`; placement may waive lessons but not the project or assessment.

## Outcomes

{evidence}

## Advanced modules

{modules}

Each module requires a concept brief, implementation lab, debugging exercise, primary-source review, and a production or research decision record. Learners maintain an experiment log and submit reusable tests rather than screenshots.

## Required specialization project

{track.project}

The project must include requirements, baselines, versioned data/model/code, automated tests, multi-dimensional evaluation, operations, responsible-use review, and a reproducible demonstration.

## Research or systems extension

{track.extension}

Publish the protocol before final results. Report compute, negative outcomes, deviations, uncertainty, and transfer limits.

## Portfolio evidence

{evidence}

## Final assessment

1. **Practical build (40%):** reproduce the project from a clean environment and pass seeded failure tests.
2. **Design defense (25%):** justify architecture, data, evaluation, cost, and rejected alternatives.
3. **Research/systems extension (20%):** defend method, evidence, uncertainty, and limitations.
4. **Operations and responsibility (15%):** execute an incident or adversarial drill and explain residual risk.

Passing requires 80/100 overall, no section below 70%, and no critical correctness, leakage, security, privacy, licensing, or reproducibility defect. Revise failed evidence and document the correction before reassessment.
"""


def main() -> None:
    target = ROOT / "specializations"
    target.mkdir(exist_ok=True)
    rows = []
    for slug, track in TRACKS.items():
        folder = target / slug
        folder.mkdir(exist_ok=True)
        (folder / "README.md").write_text(render(slug, track), encoding="utf-8")
        rows.append(f"| [{track.title}]({slug}/README.md) | {track.duration} | `{track.prerequisite}` |")
    index = """# Specializations

Complete the common core, then select at least one advanced specialization. Placement can waive study material but not competency evidence.

| Track | Duration | Entry gate |
|---|---:|---|
""" + "\n".join(rows) + "\n"
    (target / "README.md").write_text(index, encoding="utf-8")
    print(f"Built {len(TRACKS)} specialization paths.")


if __name__ == "__main__":
    main()
