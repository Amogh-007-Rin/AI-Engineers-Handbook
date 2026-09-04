<p align="center">
  <img src="assets/branding/ai-engineers-handbook-banner.webp" alt="AI Engineers Handbook — from first principles to production" width="100%">
</p>

<h1 align="center">AI Engineers Handbook</h1>

<p align="center">
  <strong>Learn the foundations. Build real systems. Prove what works.</strong>
</p>

<p align="center">
  <a href="https://github.com/Amogh-007-Rin/AI-Engineers-Handbook/actions/workflows/quality.yml"><img src="https://github.com/Amogh-007-Rin/AI-Engineers-Handbook/actions/workflows/quality.yml/badge.svg" alt="Quality workflow"></a>
  <a href="https://github.com/Amogh-007-Rin/AI-Engineers-Handbook/actions/workflows/mlops-academies.yml"><img src="https://github.com/Amogh-007-Rin/AI-Engineers-Handbook/actions/workflows/mlops-academies.yml/badge.svg" alt="MLOps academy workflow"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-F59E0B.svg" alt="MIT License"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-22C55E.svg" alt="Contributions welcome"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Curriculum-17_tracks-00D4FF" alt="17 curriculum tracks">
  <img src="https://img.shields.io/badge/Academies-60_tools-7C3AED" alt="60 tool academies">
  <img src="https://img.shields.io/badge/Specializations-10_paths-EC4899" alt="10 specialization paths">
  <img src="https://img.shields.io/badge/Approach-CPU_first-14B8A6" alt="CPU-first approach">
  <img src="https://img.shields.io/badge/Level-Beginner_%E2%86%92_Advanced-F97316" alt="Beginner to advanced">
</p>

<p align="center">
  <a href="#-start-learning"><strong>Start learning</strong></a> ·
  <a href="#-curriculum-map"><strong>Explore curriculum</strong></a> ·
  <a href="projects/README.md"><strong>Build projects</strong></a> ·
  <a href="Libraries/CATALOG.md"><strong>Browse 60 academies</strong></a> ·
  <a href="#-contributing"><strong>Contribute</strong></a>
</p>

---

A free, open-source learning system for becoming an AI engineer—from Python,
mathematics, and data foundations to machine learning, deep learning, domain AI,
generative systems, agents, research, and production operations.

This is not an API-link collection. Each learning slice targets an observable
outcome: explain the concept, implement it, test it, diagnose failure, and
produce evidence that another person can review.

> [!IMPORTANT]
> **Project maturity:** all 17 curriculum tracks and all 60 library academies have
> substantive structural learning slices. Content remains mostly `draft` or
> `review` until independent technical, pedagogical, accessibility, security,
> and learner-journey evidence is recorded. Consult the
> [verification report](reports/verification-status.md) before treating this as
> a stable reviewed release.

## ✨ What makes this different

| Principle | What it means in practice |
|---|---|
| 🧠 **Concept before framework** | Understand the invariant before learning a library abstraction. |
| 🧪 **Evidence before confidence** | Tests, baselines, failure cases, and uncertainty support every claim. |
| 🛠️ **Build to learn** | Each track leads to runnable work, not passive completion badges. |
| 🛡️ **Safety by design** | Privacy, security, misuse, fairness, and rollback are engineering requirements. |
| 🌍 **Accessible by default** | The core targets normal CPUs and free hosted accelerators. |
| 🔁 **Production is part of AI** | Serving, monitoring, incidents, maintenance, and retirement are first-class topics. |

## 🎁 What is included

- **17 concept-first tracks** from beginner foundations through production,
  research, responsible AI, and career practice.
- **60 tool academies** with foundation, practitioner, advanced, and
  maintainer-ready progression.
- **10 specializations** covering ML, deep learning, research, NLP/LLMs, vision,
  GenAI/agents, RL, MLOps, data-centric AI, and responsible AI.
- **Runnable projects and tests** with separate solutions and scored assessments.
- **A 14-step project ladder** and a transparent
  [graduation rubric](assessments/graduation-rubric.md).
- **Shared references** for [AI terminology](references/glossary.md),
  [dataset practice](datasets/README.md), and
  [reproducibility](shared/reproducibility-checklist.md).
- **Dependency-free notebooks** for short executable learning checks.
- **A production capstone** spanning data, modeling, serving, operations, safety,
  documentation, and defense.
- **Automated content contracts** for metadata, links, prerequisites, notebooks,
  project coverage, secrets, and repository hygiene.
- **Isolated CI lanes** for lightweight, NLP, visualization, framework,
  compatibility, domain, and MLOps academies.

## 👥 Who this is for

The flagship route assumes basic computer literacy, not prior professional
software or advanced mathematics experience. Experienced developers, analysts,
researchers, and ML practitioners can enter later, but placement can waive study
material—not competency evidence.

You will get the most value if you run examples, predict outputs before
execution, record failures and fixes, complete stage gates, compare tools with
simpler alternatives, and request review of finished projects.

## 🚀 Start learning

1. Read [How to learn with this handbook](curriculum/foundations/00-orientation/README.md).
2. Select an entry point from the [curriculum map](curriculum/README.md).
3. Complete the lesson, exercises, project, tests, and assessment for each slice;
   use the [assessment index](assessments/README.md) to preserve evidence.
4. Use the [library catalog](Libraries/CATALOG.md) when a project introduces a
   tool you need to study more deeply.
5. Choose a [specialization](specializations/README.md) after the common core.
6. Follow the [project ladder](projects/README.md) and finish with the
   [production AI capstone](projects/capstone/README.md).

If you are unsure where to begin, follow orientation → foundations → mathematics
→ data → machine learning → deep learning.

### Choose your route

| Your starting point | Recommended route | First proof |
|---|---|---|
| 🌱 **New to programming** | Orientation → Python/software → mathematics → data | A tested Python project and learning log |
| 💻 **Software developer** | Placement checks → mathematics/data gaps → ML systems | A leakage-safe baseline with reproducible evaluation |
| 📊 **Analyst or data practitioner** | Software foundations → ML → specialization | A packaged pipeline with tests and a model card |
| 🔬 **Research-focused learner** | Core ML/DL → research → domain specialization | A reproduced result with an ablation and limitations |
| ⚙️ **ML practitioner** | Production AI → responsible AI → capstone | A monitored service with rollback and incident evidence |

## 🗺️ Curriculum map

| Area | Primary outcome | Entry point |
|---|---|---|
| Foundations | Python, Git, testing, debugging, packaging, and learning practice | [Foundations](curriculum/foundations/README.md) |
| Mathematics | Linear algebra, calculus, probability, statistics, and optimization | [Mathematics](curriculum/mathematics/README.md) |
| Data engineering | Contracts, validation, provenance, leakage, and pipelines | [Data](curriculum/data/README.md) |
| Machine learning | Framing, baselines, evaluation, tuning, and interpretation | [Machine learning](curriculum/machine-learning/README.md) |
| Deep learning | Neural foundations and cross-framework reasoning | [Deep learning](curriculum/deep-learning/README.md) |
| Computer vision | Image pipelines, robustness, detection, and multimodal evaluation | [Computer vision](curriculum/computer-vision/README.md) |
| NLP and speech | Language/speech pipelines, representation, evaluation, and safety | [NLP and speech](curriculum/nlp-and-speech/README.md) |
| Time series | Temporal validation, forecasting, uncertainty, and monitoring | [Time series](curriculum/time-series/README.md) |
| Recommenders | Retrieval, ranking, online metrics, and feedback loops | [Recommender systems](curriculum/recommender-systems/README.md) |
| Graph learning | Relational modeling, message passing, and structural leakage | [Graph learning](curriculum/graph-learning/README.md) |
| Reinforcement learning | Bandits, MDPs, evaluation, reproducibility, and safety | [Reinforcement learning](curriculum/reinforcement-learning/README.md) |
| Generative AI | Foundation models, RAG, evaluation, inference, and risk controls | [Generative AI](curriculum/generative-ai/README.md) |
| Agents | Bounded tool use, state, planning, evaluation, and recovery | [AI agents](curriculum/agents/README.md) |
| ML systems | Serving, CI/CD/CT, observability, incidents, and rollback | [ML systems](curriculum/ml-systems/README.md) |
| Responsible AI | Privacy, fairness, security, governance, and human oversight | [Responsible AI](curriculum/responsible-ai/README.md) |
| Research | Literature review, reproduction, ablation, and evidence quality | [Research](curriculum/research/README.md) |
| Career and capstone | Portfolio, system design, collaboration, and defense | [Career](curriculum/career/README.md) |

## 🧰 Library academies

The [complete catalog](Libraries/CATALOG.md) covers 60 tools across scientific
computing, visualization, classical ML, deep learning, NLP/generative AI,
computer vision, reinforcement learning, MLOps/serving, and graph learning.

Every academy contains an opinionated mental model, a basic-to-advanced route,
exercises, runnable project code and tests, separated solutions, a scored
assessment, a declared environment, troubleshooting, and production/security
guidance. The curriculum teaches concepts; academies teach tool-specific
workflows. Use both.

## 🔄 How a learning slice works

```text
prerequisites → lesson → guided exercises → independent project
                                      ↓
                         tests + solution + assessment
                                      ↓
                         review evidence + next gate
```

Metadata records prerequisites, level, outcomes, formats, compute needs, status,
and verification date. Validation rejects invalid links, duplicate slugs,
missing prerequisites, cycles, and unsupported publication states.

## ⚡ Local setup

The dependency-free core requires Python 3.12 or newer:

```bash
git clone https://github.com/Amogh-007-Rin/AI-Engineers-Handbook.git
cd AI-Engineers-Handbook
python3 scripts/validate_content.py
python3 scripts/execute_notebooks.py
python3 scripts/run_contract_tests.py
```

Install each academy in its own environment instead of combining 60 toolchains:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Libraries/NumPy/environment/requirements.txt
python -W error -m unittest discover -s Libraries/NumPy/projects -v
```

Some frameworks require Python 3.12 because newer upstream wheels are not yet
available. Check the academy environment and
[verification report](reports/verification-status.md) before installation.
Core examples target CPUs or free hosted accelerators; costly extensions are
explicitly labeled.

## ✅ Validate the repository

Run the complete local dependency-free gate from the repository root:

```bash
python3 scripts/validate_content.py
python3 scripts/audit_library_academies.py --strict
python3 scripts/audit_curriculum.py --strict
python3 scripts/audit_release_readiness.py
python3 scripts/execute_notebooks.py
python3 scripts/run_contract_tests.py
python3 scripts/scan_repository.py
python3 -m unittest discover -s tests -v
git diff --check
```

These checks prove repository contracts and dependency-free behavior, not every
heavy framework, container, external service, or human review. Workflows under
[`.github/workflows`](.github/workflows) provide isolated CI lanes; successful
run URLs must be recorded before a stable release claim.

Run `python3 scripts/audit_release_readiness.py --strict` only when preparing a
stable tag. It will fail until the structured remote-run, review, and per-academy
learner evidence in `reports/release-evidence.json` is complete.

## 🏗️ Repository architecture

```text
AI-Engineers-Handbook/
├── curriculum/       # 17 concept-first tracks and stage gates
├── Libraries/        # 60 basic-to-advanced tool academies
├── specializations/  # 10 advanced role/domain routes
├── projects/         # cross-track work and production capstone
├── notebooks/        # dependency-free executable checks
├── schema/           # machine-readable content contract
├── scripts/          # audits, validators, runners, and hygiene checks
├── tests/            # tests for repository automation
├── templates/        # content and review templates
├── reports/          # verification and release evidence
├── project.md        # full blueprint and acceptance criteria
└── MAINTENANCE.md    # review cadence and incident workflow
```

## 🤝 Contributing

Corrections, complete learning slices, tests, accessibility improvements,
translations, compatibility fixes, learner feedback, and independent reviews
are welcome.

Before opening a pull request:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [project.md](project.md).
2. Focus the change on a learner outcome, not page count.
3. Add executable evidence when behavior changes.
4. Cite primary sources and record licenses/provenance for external assets.
5. Run the relevant academy tests and repository quality gate.
6. Describe intended learners, compute, evidence, and known limitations.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Review
and upkeep expectations are in [MAINTENANCE.md](MAINTENANCE.md).

## 🔐 Security and responsible disclosure

Do not publish credentials, private data, or an unmitigated exploit in an issue.
Follow [SECURITY.md](SECURITY.md) for private reporting. Examples should use
synthetic or appropriately licensed data, bounded side effects, redacted
secrets, and explicit artifact trust boundaries.

## 📍 Roadmap and release maturity

The blueprint and definition of done live in [project.md](project.md). A stable
release additionally requires independent domain, pedagogy, accessibility, and
security reviews; foundation and practitioner learner journeys; and clean remote
execution of heavy framework/container lanes. Green unit tests do not waive
those requirements. The [verification ledger](reports/verification-status.md)
is the authoritative record of proof and remaining work.

## 📜 License

Code and original documentation are available under the [MIT License](LICENSE).
Third-party datasets, models, papers, images, and quoted material retain their
own licenses and require separate attribution. The generated hero artwork has a
public [asset provenance record](assets/branding/README.md) with its prompt,
tooling, optimization, and accessibility notes.

## 💜 Maintainer

Initial maintainer: [@Amogh-007-Rin](https://github.com/Amogh-007-Rin).
Ownership covers triage and coordination; specialized publication gates still
require independent reviewers.
