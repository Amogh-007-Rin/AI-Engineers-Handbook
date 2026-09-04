# AI Engineers Handbook — Project Blueprint

## 1. Vision

The AI Engineers Handbook will be a free, open-source learning system that takes a learner from basic computer literacy to advanced AI engineering practice. It will combine a guided curriculum, runnable laboratories, assessments, portfolio projects, specialization tracks, and a searchable technical reference.

The handbook is intended primarily for career switchers studying for 12–18 months at roughly 10–15 hours per week. Experienced developers, data analysts, and researchers will be able to use placement guidance and accelerated paths without weakening the graduation standard.

Finishing the handbook should give learners the knowledge, practical experience, engineering habits, and portfolio expected of a strong AI engineer. Completion alone cannot guarantee employment or professional mastery; those also require sustained practice, judgment, collaboration, and real-world experience.

### Success criteria

A successful graduate can:

- Explain the mathematics, statistics, and computer-science ideas underlying modern AI.
- Implement important algorithms from first principles and use production libraries responsibly.
- Select methods based on the problem, data, evidence, compute budget, and operational constraints.
- Build, evaluate, debug, deploy, monitor, secure, and improve complete AI systems.
- Work competently with PyTorch, TensorFlow, and JAX.
- Build classical ML, deep-learning, multimodal, retrieval-augmented generation, agentic, and distributed AI applications.
- Read research papers, design controlled experiments, run ablations, and reproduce selected results.
- Analyze accuracy, reliability, latency, cost, privacy, fairness, security, accessibility, and environmental tradeoffs.
- Communicate uncertainty, document decisions, review code, and collaborate effectively.
- Present a reviewed portfolio and a production-grade capstone.

### Non-goals

The project will not:

- Promise employment or expertise merely for reading the material.
- Duplicate complete official API documentation.
- Copy copyrighted courses, books, diagrams, datasets, or exercises.
- Require paid infrastructure for the core curriculum.
- Publish large collections of empty lesson placeholders and call them a release.

## 2. Learning Model

The handbook will use five connected layers:

1. **Curriculum:** A prerequisite-driven beginner-to-advanced path.
2. **Specializations:** Advanced role- and domain-focused study after the common core.
3. **Practice and assessment:** Exercises, labs, projects, examinations, paper reproductions, and a capstone.
4. **Library academies:** Standalone basic-to-advanced learning paths for every tool in `Libraries/`.
5. **Reference:** Concise, searchable guides for terminology and operational tasks.

Every curriculum stage follows the same learning loop:

> Learn → implement → experiment → evaluate → build → explain → reflect

Learners should not advance merely because they opened every lesson. Each stage has explicit evidence of mastery and remediation guidance.

## 3. Target Repository Architecture

```text
AI-Engineers-Handbook/
├── README.md                 # Learner-facing landing page
├── project.md                # This project blueprint
├── curriculum/               # The shared, prerequisite-driven core
│   ├── foundations/
│   ├── mathematics/
│   ├── data/
│   ├── machine-learning/
│   ├── deep-learning/
│   ├── computer-vision/
│   ├── nlp-and-speech/
│   ├── reinforcement-learning/
│   ├── generative-ai/
│   ├── agents/
│   ├── ml-systems/
│   ├── responsible-ai/
│   ├── research/
│   └── career/
├── specializations/          # Advanced role and domain paths
├── projects/                 # Integrated, progressively independent work
├── assessments/              # Knowledge checks, exams, rubrics, solutions
├── Libraries/                # Basic-to-advanced library academies
├── references/               # Tool and concept reference material
├── datasets/                 # Dataset cards and download helpers, not large data
├── shared/                   # Reusable code, diagrams, templates, and glossary
└── contributing/             # Authoring, review, governance, and maintenance
```

All new curriculum paths use lowercase kebab-case. Directory entry points use `README.md`. Existing names under `Libraries/` remain stable during the first content pass to avoid disruptive bulk renames; a later migration may normalize them with redirects. The library folders must not be mass-deleted or replaced with another set of placeholders.

The root README will eventually provide:

- A clear “Start here” route.
- Environment setup and troubleshooting.
- A visual curriculum map.
- Placement guidance and accelerated routes.
- Links to specializations, projects, references, contribution guidance, and progress tracking.

## 4. Flagship Curriculum

The curriculum is organized by competency rather than by library. Exact lesson counts are decided while designing each vertical milestone, but the outcomes below are mandatory.

### Stage 0 — Orientation and learning practice

- Understand the curriculum, mastery gates, expected workload, and support model.
- Learn effective technical study, deliberate practice, note-taking, debugging, and asking good questions.
- Set up a reproducible local environment and an optional free hosted-notebook environment.
- Learn responsible use of AI coding assistants without outsourcing understanding.

**Gate:** Complete the environment diagnostic, study-plan exercise, and a small command-line task.

### Stage 1 — Software engineering foundations

- Python syntax, data structures, functions, object-oriented and functional patterns, typing, exceptions, iterators, concurrency basics, and performance awareness.
- Terminal, Linux fundamentals, Git/GitHub, dependency management, packaging, debugging, logging, unit/integration testing, documentation, and code review.
- SQL, relational modeling, HTTP, REST, serialization, command-line interfaces, and API consumption.
- Core algorithms, data structures, complexity, and clean-code principles relevant to AI workloads.

**Gate:** Build, test, document, and package a small data-oriented Python application.

### Stage 2 — Mathematics and statistics for AI

- Algebra, functions, vectors, matrices, tensors, linear transformations, decompositions, and numerical stability.
- Derivatives, gradients, Jacobians, optimization, constrained optimization, and automatic differentiation.
- Probability, common distributions, Bayes’ rule, expectation, variance, sampling, and probabilistic modeling.
- Descriptive and inferential statistics, confidence intervals, hypothesis tests, effect size, power, and multiple comparisons.
- Information theory, experiment design, causal-inference foundations, and numerical methods.

Every major topic must include visual intuition, a derivation at the appropriate level, implementation, and an AI application.

**Gate:** Pass a mixed conceptual/computational exam and complete a statistical investigation with reproducible conclusions.

### Stage 3 — Data foundations

- NumPy, Pandas, Polars, visualization, SQL, data collection, schemas, cleaning, missingness, outliers, and validation.
- Exploratory analysis, feature engineering, sampling, class imbalance, dataset shift, leakage, annotation quality, and data-centric iteration.
- Dataset versioning, lineage, privacy, licenses, datasheets, and reproducible pipelines.
- Larger-than-memory and distributed data fundamentals with Dask, Spark, and Ray where justified.

**Gate:** Produce a tested data pipeline, dataset card, exploratory report, and leakage review.

### Stage 4 — Classical machine learning

- Problem framing, baseline design, train/validation/test strategy, cross-validation, metrics, calibration, and error analysis.
- Linear and generalized linear models, nearest neighbors, naive Bayes, decision trees, random forests, gradient boosting, kernels, and ensembles.
- Clustering, dimensionality reduction, anomaly detection, recommendation foundations, and time-series forecasting.
- Feature selection, regularization, hyperparameter optimization, interpretability, uncertainty, and causal-inference foundations.
- Scikit-learn as the primary teaching interface, with XGBoost, LightGBM, CatBoost, Statsmodels, Optuna, and SHAP introduced for justified use cases.

**Gate:** Build and defend an end-to-end ML system with baselines, controlled experiments, error analysis, model card, and reproducible results.

### Stage 5 — Deep-learning foundations

- Tensors, computation graphs, automatic differentiation, initialization, optimization, normalization, regularization, and training diagnostics.
- Multilayer perceptrons, convolutional networks, sequence models, attention, transformers, graph neural networks, autoencoders, GANs, and diffusion foundations.
- Efficient input pipelines, mixed precision, profiling, distributed training, checkpointing, and reproducibility.
- Framework-native implementation and debugging in PyTorch, TensorFlow, and JAX.

**Gate:** Implement a neural network and backpropagation from scratch, then complete a controlled cross-framework benchmark.

### Stage 6 — Computer vision and multimodal AI

- Image formation and processing, augmentation, classification, detection, segmentation, keypoints, tracking, and video.
- CNNs, vision transformers, self-supervised vision, generative imaging, and vision-language models.
- Dataset bias, robustness, evaluation, optimization, and edge/production deployment.

**Gate:** Complete a vision system with task-appropriate metrics, failure analysis, deployment constraints, and an ethical-risk assessment.

### Stage 7 — NLP and speech

- Text normalization, tokenization, linguistic processing, sparse representations, embeddings, and classical NLP.
- Sequence models, attention, transformers, language modeling, information extraction, retrieval, translation, and summarization.
- Speech recognition, speech synthesis, multilingual systems, evaluation, bias, and production constraints.

**Gate:** Build and evaluate an NLP or speech application against a meaningful baseline and documented failure taxonomy.

### Stage 8 — Reinforcement learning and decision systems

- Bandits, Markov decision processes, dynamic programming, Monte Carlo methods, and temporal-difference learning.
- Value-based, policy-gradient, actor-critic, offline, model-based, and deep RL.
- Multi-agent systems, exploration, reward design, evaluation, reproducibility, sim-to-real concerns, and safety.

**Gate:** Implement core algorithms, evaluate across multiple seeds, and document instability and reward-design risks.

### Stage 9 — Generative AI and foundation models

- Tokenization, transformer internals, pretraining objectives, scaling concepts, alignment methods, and inference.
- Prompt and context design, structured outputs, tool calling, embeddings, vector search, retrieval-augmented generation, and reranking.
- Fine-tuning, parameter-efficient tuning, distillation, quantization, serving, caching, and cost/latency optimization.
- Synthetic data, multimodal generation, automated and human evaluation, red teaming, and failure analysis.

**Gate:** Deliver an evaluated RAG or adapted-model application whose claims are supported by retrieval, generation, safety, latency, and cost measurements.

### Stage 10 — AI agents

- Deterministic workflows versus autonomous agents, tool use, planning, memory, state, delegation, and human approval.
- Context management, interoperability protocols, sandboxing, permissions, prompt injection, and data exfiltration risks.
- Observability, traces, offline/online evaluation, recovery, idempotency, budgets, and failure containment.
- Single-agent and multi-agent patterns, including when not to use an agent.

**Gate:** Build a tool-using agent with an explicit threat model, deterministic evaluation suite, approval boundaries, and recovery tests.

### Stage 11 — Production AI and MLOps

- Reproducibility, experiment tracking, data/model versioning, registries, pipelines, orchestration, and lineage.
- Batch, streaming, online, and edge inference; FastAPI; model servers; containers; Kubernetes; and autoscaling.
- CI/CD/CT, testing ML systems, shadow and canary releases, monitoring, drift, rollback, incident response, and service-level objectives.
- Feature stores, distributed compute, hardware awareness, profiling, reliability, security, privacy, cost, and carbon awareness.
- System design for recommendation, search, forecasting, computer vision, LLM, RAG, and agent workloads.

**Gate:** Deploy and operate a versioned AI service with automated tests, monitoring, rollback, load results, runbook, and postmortem exercise.

### Stage 12 — Responsible and secure AI

- Fairness, explainability, transparency, accountability, accessibility, privacy, and human-centered design.
- Adversarial ML, data poisoning, model theft, model and dependency supply-chain risks, prompt injection, and abuse prevention.
- Governance, provenance, model/data cards, licensing, intellectual property, regulatory awareness, and environmental impact.
- Risk classification, threat modeling, red teaming, incident escalation, and human oversight.

Responsible-AI requirements are embedded throughout earlier projects as well as taught here directly.

**Gate:** Complete an impact assessment, threat model, red-team report, and mitigation plan for an existing project.

### Stage 13 — Research and frontier practice

- Search and review literature, read papers critically, trace claims to evidence, and identify open questions.
- Form hypotheses, define baselines, control variables, run ablations, quantify uncertainty, and report negative results.
- Reproduce a paper within declared compute limits and explain deviations from the published result.
- Write technical reports, research artifacts, and reproducibility documentation.

**Gate:** Submit a reviewed paper reproduction and an original extension or ablation.

### Stage 14 — Professional practice and capstone

- AI product thinking, requirements discovery, architecture decisions, estimation, stakeholder communication, and technical leadership.
- Code review, open-source collaboration, system-design interviews, ML interviews, portfolio writing, and communicating uncertainty.
- Design and deliver the final production-grade capstone.

**Gate:** Pass the capstone review and portfolio defense using the published graduation rubric.

## 5. Specialization Tracks

After the common core, learners select at least one track:

- Machine-learning engineer
- Deep-learning engineer
- AI research engineer
- NLP, speech, and LLM engineer
- Computer-vision and multimodal engineer
- Generative-AI and agent engineer
- Reinforcement-learning engineer
- MLOps and AI-platform engineer
- Data-centric AI engineer
- Responsible-AI, security, and evaluation engineer

Each specialization must define:

- Entry prerequisites and expected duration.
- Competency-based learning outcomes.
- Required and optional modules.
- One substantial domain project.
- A research or systems extension.
- A final assessment and public portfolio artifact.

Accelerated bridge paths may exempt lessons but not competency gates. Placement diagnostics will be available for software developers, data analysts, mathematicians, and existing ML practitioners.

## 6. Content Contracts

### Lesson contract

Markdown is canonical for explanations. Jupyter notebooks are used for exploratory and mathematical labs; normal Python packages or services are used when software structure, testing, or deployment matters.

Every published lesson contains:

- Level, estimated time, prerequisites, and measurable learning objectives.
- A conceptual explanation with intuition and appropriate mathematics.
- At least one worked example and one runnable lab.
- Common misconceptions and debugging guidance.
- Real-world and production relevance.
- Exercises covering recall, implementation, analysis, and extension.
- A knowledge check and explicit completion criteria.
- A summary, glossary additions, and authoritative further reading.
- CPU/free-tier instructions and clearly marked optional GPU extensions.
- Alternative text for meaningful visuals and accessible non-color-only notation.
- Deterministic seeds and expected outputs where practical.

### Metadata interface

Curriculum documents use concise YAML front matter:

```yaml
title: Gradient Descent
slug: gradient-descent
level: foundation
stage: mathematics
estimated_hours: 4
prerequisites:
  - derivatives
learning_objectives:
  - Implement batch gradient descent from scratch
formats:
  - lesson
  - notebook
compute: cpu
status: draft
last_verified: YYYY-MM-DD
```

Allowed content statuses are `outline`, `draft`, `review`, `published`, and `maintenance`. A validator will reject duplicate slugs, missing prerequisites, prerequisite cycles, unsupported status values, and published content missing required fields.

### Project contract

Every project defines:

- The learner brief, prerequisites, constraints, and deliverables.
- Starter assets without completed core logic.
- A public rubric and explicit pass criteria.
- Testing and reproducibility requirements.
- Dataset source, license, limitations, and download process.
- Baseline results or behavioral acceptance ranges.
- Security, privacy, accessibility, and responsible-AI considerations appropriate to the project.
- A separated reference solution and instructor notes.

Solutions must be helpful to self-directed learners without appearing inline before a genuine attempt is expected.

## 7. Framework Parity

PyTorch, TensorFlow, and JAX are equal first-class frameworks in the deep-learning core.

- Teach shared theory once, followed by framework-native implementations.
- Use equivalent datasets, objectives, metrics, and expected outputs for comparative labs.
- Explain each framework’s execution, state, compilation, and debugging model instead of mechanically translating syntax.
- Require learners to complete one implementation, compare a second, and periodically reproduce a model in all three.
- Maintain automated smoke tests for all three implementations.
- Use independent pinned environments when a shared environment would be unstable.
- Allow framework-specific advanced material where artificial parity would reduce educational value.

Framework parity applies to core competencies, not identical page or line counts.

## 8. Library Academies

The `Libraries/` module will provide a complete, independent learning path for every included library, framework, platform, or method. These paths complement the concept-first curriculum: the curriculum answers **what to learn and why**, while a library academy teaches **how to use one tool well from first contact through production practice**.

All 60 current library/tool/method entries are in scope:

- **Data, scientific computing, and visualization:** NumPy, Pandas, SciPy, Statsmodels, Polars, DuckDB, Dask, PySpark, Matplotlib, Seaborn, Plotly, Bokeh, Dash, and NetworkX.
- **Classical ML, optimization, and forecasting:** Scikit-Learn, XGBoost, LightGBM, CatBoost, PyCaret, Optuna, Prophet, Orbit, and ARIMA-SARIMA.
- **Deep learning and interoperability:** PyTorch, TensorFlow, Keras, JAX, Flax, FastAI, ONNX, PyTorch Geometric, and DGL.
- **NLP, generative AI, and conversational systems:** NLTK, spaCy, Gensim, SentenceTransformers, HuggingFace Transformers, HuggingFace Diffusers, OpenAI SDK, and Rasa.
- **Computer vision:** OpenCV, Albumentations, Ultralytics YOLO, MMDetection, Detectron2, and DeepFace.
- **Reinforcement learning:** Gymnasium, Stable-Baselines3, RLlib, and PettingZoo.
- **Production, orchestration, and MLOps:** FastAPI, MLflow, Weights & Biases, BentoML, TensorFlow Serving, TorchServe, Ray, Airflow, Docker, and Kubernetes.

`Z-Roadmap` is migration material rather than a library academy and will become part of the generated curriculum roadmap. `ARIMA-SARIMA` is a technique-focused academy; it follows the same learning contract but does not pretend to be a software library.

### 8.1 Standard academy structure

Every library folder will converge on this structure, omitting only files that genuinely do not apply:

```text
Libraries/<Library Name>/
├── README.md                 # Overview, prerequisites, map, and progress checklist
├── 00-setup/
├── 01-fundamentals/
├── 02-core-workflows/
├── 03-intermediate/
├── 04-advanced/
├── 05-production/
├── 06-internals/             # Architecture/performance internals where useful
├── cookbook/                 # Focused, searchable recipes
├── projects/                 # Guided and independent projects
├── assessments/              # Checks, exercises, rubrics, and separated solutions
├── troubleshooting/          # Common errors, diagnostics, and compatibility notes
└── environment/              # Pinned requirements and reproducibility instructions
```

Small or narrow tools may combine adjacent levels, but they must preserve the same learning progression and exit criteria. Large ecosystems may add domain tracks beneath a level. Empty directories will not be committed merely to match the template.

### 8.2 Required learning progression

Each academy teaches the following progression:

1. **Orientation and setup**
   - What the tool is, when to use it, when not to use it, alternatives, prerequisites, installation, environment verification, and a five-minute first success.
2. **Fundamentals**
   - Mental model, core abstractions, essential API surface, data structures, lifecycle, and small worked examples.
3. **Core workflows**
   - The common end-to-end tasks practitioners perform, with realistic data and tests rather than disconnected syntax demonstrations.
4. **Intermediate practice**
   - Composition, configuration, reusable components, evaluation, debugging, serialization, integrations, and common design patterns.
5. **Advanced practice**
   - Customization, extension APIs, scaling, profiling, optimization, distributed or accelerator use where relevant, and non-obvious failure modes.
6. **Production use**
   - Testing, reproducibility, observability, security, deployment, versioning, performance, cost, migration, and operational troubleshooting as applicable.
7. **Internals and expert judgment**
   - Architecture, execution model, tradeoffs, limitations, source-code navigation, contribution workflow, and selecting the tool against credible alternatives.
8. **Mastery evidence**
   - Knowledge checks, implementation exercises, one guided project, one independent project, and an exit assessment mapped to stated outcomes.

### 8.3 Academy README contract

Each `Libraries/<Library Name>/README.md` becomes the academy landing page and must include:

- Purpose, current support status, audience, prerequisites, and estimated study time.
- Version and environment policy, installation verification, and compute requirements.
- A basic-to-advanced module map with links and progress checkboxes.
- Learning outcomes for foundation, practitioner, and advanced completion levels.
- “Use it when / avoid it when” guidance and a comparison with realistic alternatives.
- Links into relevant concept-first curriculum lessons and upstream official documentation.
- Project and assessment requirements.
- Known compatibility, security, licensing, and maintenance notes.
- `last_verified` metadata and the owners/reviewers responsible for maintenance.

The README is navigation and orientation, not a single oversized tutorial. Substantive teaching belongs in focused lessons and runnable artifacts.

### 8.4 Library lesson contract

Library lessons follow the global lesson contract and additionally require:

- A pinned, documented version or supported version range.
- Expected output and an executable verification step.
- Explanation of the underlying concept, not only copy-paste API calls.
- At least one misuse, failure mode, or debugging example.
- Links to the exact upstream documentation relevant to the lesson.
- A note separating stable concepts from version-sensitive syntax.
- Tests for reusable code and clean notebook execution for exploratory material.
- A migration note when the recommended API changed across supported versions.

Generated API inventories are allowed as supplements but cannot replace authored explanations, workflows, or exercises.

### 8.5 Tailoring by library family

The common template will be adapted rather than applied mechanically:

- **Data and visualization tools:** emphasize data models, indexing, correctness, missing data, performance, memory, interoperability, and truthful visual communication.
- **ML libraries:** emphasize problem framing, pipelines, leakage prevention, evaluation, tuning, interpretability, persistence, and reproducibility.
- **Deep-learning frameworks:** emphasize tensors, autodiff, modules, input pipelines, training loops, distributed execution, compilation, profiling, export, and debugging.
- **NLP/CV/RL libraries:** emphasize domain data formats, preprocessing, pretrained assets, task metrics, fine-tuning, failure analysis, and domain-specific risks.
- **Serving and MLOps tools:** emphasize lifecycle, configuration, local-to-production workflow, observability, reliability, security, upgrades, rollback, and incident handling.
- **Infrastructure tools:** teach only the subset required for robust AI systems before advanced platform administration.
- **Hosted SDKs:** isolate provider-specific material, mock network calls in core tests, teach cost and privacy controls, and never require learner-owned paid credentials.

### 8.6 Projects and assessments

Each academy includes at minimum:

- A quick-start exercise that proves the environment works.
- Short checks after each level.
- Debugging tasks based on realistic mistakes.
- A guided project that demonstrates the canonical workflow.
- An independent project with a public rubric and multiple valid solutions.
- A final practical assessment covering selection, implementation, debugging, performance, and tradeoff reasoning.

Projects should connect across libraries when that reflects real practice—for example, Pandas → Scikit-Learn → MLflow → FastAPI → Docker—but each academy must state precisely which competencies it owns and which it reuses.

### 8.7 Completion levels

Library progress is reported using evidence-based levels:

- **Foundation:** Can install the tool, explain its mental model, and complete essential tasks with documentation.
- **Practitioner:** Can build reliable end-to-end workflows, test them, debug common failures, and choose appropriate APIs.
- **Advanced:** Can optimize, extend, integrate, and operate the tool while explaining tradeoffs and limitations.
- **Maintainer-ready:** Optional level for reading source, diagnosing internals, contributing fixes, and managing upgrades; it is not required for handbook graduation.

Completion badges, if introduced, must link to assessment evidence and must not be awarded for page views.

### 8.8 Development waves

Library academies will be authored in dependency-aware waves, with each selected academy completed vertically before the next wave expands:

1. **Core foundation:** NumPy, Pandas, Matplotlib, Seaborn, SciPy, Statsmodels, Polars, DuckDB, and FastAPI.
2. **Classical ML:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna, PyCaret, ARIMA-SARIMA, Prophet, and Orbit.
3. **Deep learning:** PyTorch, TensorFlow, Keras, JAX, Flax, FastAI, and ONNX.
4. **Domain AI:** OpenCV, Albumentations, Transformers, Diffusers, NLTK, spaCy, Gensim, SentenceTransformers, PyTorch Geometric, DGL, and NetworkX.
5. **Specialized systems:** YOLO, MMDetection, Detectron2, DeepFace, Rasa, Gymnasium, Stable-Baselines3, RLlib, and PettingZoo.
6. **Scale and production:** MLflow, Weights & Biases, BentoML, TensorFlow Serving, TorchServe, Ray, Dask, PySpark, Airflow, Docker, Kubernetes, Dash, Bokeh, Plotly, and OpenAI SDK.

Wave membership controls authoring order, not learner difficulty in every case. Security-sensitive or fast-changing SDK content receives a shorter verification interval.

### 8.9 Library academy definition of done

The final evidence gate is executable:

```bash
python3 scripts/audit_library_academies.py --strict
```

It requires every academy to contain a navigation guide, metadata-bearing
lesson, exercises, executable project, project tests, solution, scored
assessment, and reproducible environment. The non-strict report supports work
in progress; only the strict report supports a completion claim.

An academy is complete only when:

- Its README contract and all mandatory progression levels are satisfied.
- Every published lesson has valid metadata, working links, citations, and version notes.
- Core examples and notebooks execute in a clean, pinned environment.
- Exercises, guided project, independent project, assessment, rubrics, and separated solutions are present.
- CPU/free-tier requirements are documented and verified.
- The academy links to prerequisite concepts without needlessly duplicating curriculum content.
- Security, licensing, accessibility, and responsible-use checks are complete.
- At least one foundation-level and one practitioner-level learner journey has been tested.
- A domain reviewer approves technical depth and a second reviewer approves pedagogy.
- CI passes and ongoing ownership is assigned.

## 9. Compute and Environment Policy

- Every core exercise must run on a normal CPU or a free hosted GPU within a documented time limit.
- Paid compute, multi-GPU training, and proprietary services are optional extensions.
- Expensive examples use small datasets, reduced models, cached artifacts, or simulation while preserving the target concept.
- Each runnable artifact declares approximate RAM, storage, runtime, accelerator, and network requirements.
- Environments are pinned and reproducible. Secrets are loaded through documented environment mechanisms and never committed.
- Large datasets, checkpoints, and generated artifacts are downloaded or stored externally with checksums and licenses.

## 10. Assessment System

Assessment is layered rather than dependent on a single examination:

- Lesson-level knowledge checks.
- Deterministic, autograded coding exercises where suitable.
- Manual rubrics for design, experimentation, research, and communication.
- Stage-end practical examinations.
- At least one substantial project per major stage.
- Paper reproductions with baselines, methodology, results, deviations, and retrospectives.
- Foundational, specialization, and capstone-ready portfolio reviews.
- A production-grade final capstone.

Rubrics score:

- Conceptual and technical correctness.
- Reasoning and justified tradeoffs.
- Reproducibility and experimental discipline.
- Code quality, tests, and documentation.
- Data and evaluation quality.
- Reliability, security, privacy, and responsible design.
- System architecture, operations, cost, and performance.
- Written, visual, and verbal communication.

Every assessment publishes pass criteria, expected evidence, common failure modes, and remediation. Learners may revise and resubmit; mastery matters more than first-attempt performance.

## 11. Project Ladder

Projects become progressively less guided:

1. Python data explorer.
2. Reproducible statistical investigation.
3. Tested data pipeline and dataset card.
4. End-to-end tabular ML system.
5. Neural network and backpropagation from scratch.
6. Cross-framework deep-learning benchmark.
7. Vision or NLP application.
8. Recommender, forecasting, graph, or reinforcement-learning system.
9. Evaluated RAG application.
10. Tool-using agent with safety and recovery tests.
11. Scalable model service with monitoring.
12. Research-paper reproduction.
13. Specialization project.
14. Production-grade capstone.

The capstone requires a proposal, stakeholder and risk analysis, architecture decision record, reproducible repository, data and model cards, tests, offline and online evaluation plan, deployment, monitoring, threat model, cost analysis, demo, operations runbook, and retrospective.

## 12. Editorial Quality and Governance

### Source policy

- Prefer primary sources, peer-reviewed papers, official documentation, standards, and reputable textbooks.
- Cite claims close to where they are used.
- Distinguish established facts, current evidence, simplifying teaching assumptions, opinions, and original work.
- Do not fabricate citations, benchmarks, quotations, or experimental results.
- Do not copy tutorials, books, diagrams, datasets, or exercises without compatible permission and attribution.
- Review external resources for authority, accessibility, stability, and licensing.

### Publication reviews

A lesson cannot become `published` until it passes:

1. Technical correctness review.
2. Pedagogical and prerequisite review.
3. Executability and reproducibility review.
4. Accessibility and editorial review.
5. Security and responsible-AI review where applicable.

Domain maintainers own reviews and maintenance. `CODEOWNERS` or an equivalent ownership map should be introduced when the contributor base supports it.

### Maintenance

- Record dependency versions and `last_verified` dates.
- Run scheduled link, dependency, notebook, and factual reviews.
- Mark temporarily stale content as `maintenance` rather than silently presenting it as current.
- Label issues for content gaps, beginner friction, correctness, stale material, accessibility, reproducibility, licensing, and security.
- Use learner feedback and assessment failure patterns to prioritize improvements.

## 13. Automation and Quality Gates

Automation will be introduced incrementally and will eventually check:

- Markdown style, spelling allowlists, internal links, and required front matter.
- Curriculum graph integrity, including missing and circular prerequisites.
- Notebook execution, deterministic checks, and output hygiene.
- Python formatting, linting, type checking, unit tests, and integration tests.
- Environment reproducibility across supported Python versions.
- PyTorch, TensorFlow, and JAX smoke tests.
- Dataset and model-download isolation.
- Secret detection, dependency vulnerability checks, and license policy.
- Published-content requirements and overdue verification dates.

Pull-request checks must be CPU-compatible and reasonably fast. Expensive integrations and accelerator tests run on a schedule or through explicitly triggered workflows. CI must never require contributor-owned cloud credentials.

## 14. Current Repository Migration

The current repository is a useful topic inventory but not yet a curriculum. Most of its 60 library/tool/method folders contain the same short placeholder index, the FastAPI course path is an empty tracked file, and the PDF roadmap sequences tools without lessons or mastery gates.

Migration will therefore proceed deliberately:

1. Inventory every existing folder by category, prerequisites, curriculum relationships, support status, and maintenance risk.
2. Map every academy to the competency-based curriculum and its dependency wave.
3. Keep substantial tool learning resources in `Libraries/`; use `references/` only for concise cross-cutting lookup material.
4. Replace placeholder pages with one complete academy at a time—never with a larger empty scaffold.
5. Correct the FastAPI path and integrate it into production-serving lessons during the relevant milestone.
6. Rebuild the PDF roadmap as maintainable Markdown generated from curriculum metadata.
7. Update the root README after the first complete vertical slice can give learners a working “Start here” experience.
8. Preserve inbound links with migration notes or redirects where the hosting platform permits them.
9. Remove obsolete paths only after content migration and link validation.

## 15. Vertical Delivery Milestones

Development ships complete learn-practice-assess slices. A milestone is complete only when its lessons, labs, assessments, solutions, navigation, CI checks, and high-priority learner-feedback fixes are present.

### Milestone 0 — Project foundations

- Ratify this blueprint and define contribution governance.
- Add content, notebook, project, dataset-card, model-card, and rubric templates.
- Implement the metadata schema and curriculum graph validator.
- Add baseline Markdown, link, secret, and Python checks.
- Audit existing content and create the migration inventory.
- Ratify the library academy template, metadata extension, version-support policy, and definition of done.
- Select the first foundation academies and map their lessons to Stages 1–3.
- Establish a lightweight learner-testing process and definition of done.

### Milestone 1 — Beginner-ready foundation

- Deliver orientation, environment setup, Python, terminal, Git, SQL, software engineering, math essentials, and data fundamentals.
- Add placement diagnostics, glossary, troubleshooting, progress tracking, and beginner navigation.
- Publish the first knowledge checks, practical exams, solutions, and portfolio project.
- Publish the first complete library academies needed by this path, beginning with NumPy and Pandas, without duplicating shared theory.
- Validate the entire route with fresh learners before expanding it.

### Milestone 2 — Classical machine learning

- Deliver problem framing, model families, evaluation, experimentation, interpretation, and data-centric iteration.
- Publish the classical-ML examination and end-to-end project.
- Require reproducible baselines, leakage checks, error analysis, model cards, and written conclusions.

### Milestone 3 — Deep learning across three frameworks

- Deliver deep-learning foundations and framework-native PyTorch, TensorFlow, and JAX paths.
- Add framework-isolated environments and parity smoke tests.
- Publish the from-scratch neural-network exercise, cross-framework benchmark, practical exam, and remediation material.

### Milestone 4 — Domain intelligence

- Deliver computer vision, NLP/speech, time-series, recommendation, graph learning, and reinforcement-learning foundations.
- Publish domain projects with consistent experimental and responsible-AI requirements.
- Add domain-specific specializations only after their shared foundations are usable.

### Milestone 5 — Generative AI and agents

- Deliver foundation-model concepts, prompting, RAG, adaptation, multimodality, inference optimization, agents, and evaluations.
- Embed security, red teaming, cost, latency, and human-approval requirements.
- Publish an evaluated RAG system and a safety-tested agent project.

### Milestone 6 — Production, research, and capstone

- Deliver MLOps, distributed systems, secure deployment, observability, responsible AI, and research practice.
- Publish paper-reproduction and production-service projects.
- Complete specialization contracts, capstone materials, portfolio review, and graduation rubric.

### Milestone 7 — Stable world-class release

- Close prerequisite and coverage gaps found through learner testing.
- Execute all published CPU paths from clean environments.
- Sample-test free-GPU paths and scheduled framework suites.
- Audit accessibility, citations, licensing, secrets, dependencies, and security.
- Verify all published content and resolve high-severity learner feedback.
- Complete and review all 60 library academies or explicitly mark unsupported tools with a documented replacement/migration decision.
- Publish a versioned stable release and maintenance schedule.

## 16. Milestone Definition of Done

The curriculum-wide structural gate is:

```bash
python3 scripts/audit_curriculum.py --strict
```

It requires each concept track to contain a substantive guide and metadata
lesson, executable project and tests, separated solution, and scored stage
assessment. Passing this gate is necessary but does not replace reviewer,
learner, clean-environment, licensing, or accessibility evidence.

A vertical milestone is complete when:

- All mandatory learning outcomes are covered by reviewed material.
- Every lesson has valid metadata and working prerequisite navigation.
- Labs run in documented clean environments.
- Assessments measure the stated outcomes and include remediation.
- Projects have tested starters, separated solutions, and reviewed rubrics.
- Internal links, citations, licenses, and dataset/model sources pass review.
- Core work fits the documented CPU/free-GPU budget.
- Security, privacy, accessibility, and responsible-AI requirements are addressed.
- At least one fresh learner has completed the path and blocking feedback is resolved.
- CI passes and maintainers are assigned for published material.
- Every library used in the milestone meets the library academy definition of done for the required completion level.

## 17. Immediate Implementation Backlog

Work begins with Milestone 0 in this order:

1. Create the content metadata JSON Schema and a curriculum-graph validator.
2. Create lesson, notebook, project, assessment, dataset-card, model-card, and solution templates.
3. Add authoring standards, review checklists, contributor workflow, and definition of done.
4. Generate an inventory report for the 60 current academy folders and the existing roadmap.
5. Create the library academy README, lesson, cookbook, troubleshooting, project, and assessment templates.
6. Establish CI for Markdown, links, metadata, secrets, and lightweight Python/notebook tests.
7. Design the Stage 0–3 prerequisite graph and placement diagnostics.
8. Build the first complete orientation-to-Python vertical slice.
9. Build NumPy and Pandas as the pilot academies and validate their cross-links to the curriculum.
10. Test both the curriculum slice and pilot academies with beginner feedback before expanding the foundation milestone.

No later-stage content should be mass-scaffolded while these authoring and quality foundations remain unvalidated.

## 18. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scope becomes unmaintainable | Ship vertical milestones, enforce ownership, and publish only completed slices. |
| Framework parity creates excessive duplication | Share theory and tests while keeping framework-native implementation modules. |
| Content becomes outdated | Pin environments, track verification dates, schedule maintenance, and mark stale pages. |
| Beginners are overwhelmed | Use prerequisites, staged navigation, placement tests, glossary, remediation, and learner testing. |
| Advanced coverage becomes shallow | Require projects, paper reproductions, system design, experiments, and specialization assessments. |
| Compute prevents participation | Keep the core CPU/free-GPU compatible and label paid-compute extensions. |
| Metrics encourage checkbox completion | Gate stages on evidence and public rubrics, not page completion. |
| Community contributions reduce consistency | Require templates, automated checks, domain review, and a published content contract. |
| Tool pages dominate conceptual learning | Keep the curriculum concept-first and make academies link to shared theory instead of redefining it. |
| Library academies duplicate curriculum | Teach shared concepts in the curriculum; academies own tool-specific workflows and link to prerequisites. |
| 60 academies become shallow or stale | Build in dependency waves, enforce academy completion gates, and assign versioned ownership. |

## 19. Project Defaults

- The canonical content language is English. Internationalization can begin after the stable English core.
- Python is primary; SQL, shell, JavaScript/TypeScript, and systems concepts appear where professionally relevant.
- PyTorch, TensorFlow, and JAX receive equal status in core deep-learning competencies.
- Core content is vendor-neutral and open-source-first. Vendor and cloud modules are optional.
- Core exercises use CPUs or free GPU tiers; paid compute is never a graduation requirement.
- The repository remains under its existing MIT license unless legal review determines otherwise.
- The initial product is a repository-based curriculum, not a custom learning-management platform.
- Solutions remain available to self-directed learners but are separated from learner-facing exercises.
- Large datasets and model weights are not committed to Git.
- Vertical quality takes priority over superficial breadth.
- `Libraries/` is the canonical home for basic-to-advanced tool learning paths; `references/` is reserved for concise lookup material.

## 20. Acceptance Criteria for This Blueprint

This blueprint is ready to guide implementation when it:

- Defines the audience, vision, constraints, outcomes, and limits.
- Covers all major AI disciplines through the common core or a named specialization.
- Gives beginners a prerequisite-free entry point and experienced learners placement options.
- Defines content, project, metadata, framework, compute, assessment, and publication contracts.
- Defines a reusable academy structure, progression, rollout order, and completion gate for all libraries.
- Makes graduation depend on demonstrable work.
- Provides a safe migration path for the existing repository.
- Specifies testable milestone completion rules.
- Leaves Milestone 0 implementers with an ordered, decision-complete backlog.
