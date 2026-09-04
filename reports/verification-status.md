# Verification status

Last updated: 2026-09-04.

This report distinguishes evidence currently present in the repository from
review or execution that still requires an external environment or people. It
must be updated before a stable release claim.

## Verified locally

- `python3 scripts/audit_library_academies.py --strict`: 60/60 academies contain
  substantive guides, metadata lessons, exercises, executable project code,
  project tests, separated solutions, scored assessments, and environments.
- `python3 scripts/validate_content.py`: 125 metadata documents; schema, internal
  links, unique slugs, and prerequisite graph pass.
- `python3 scripts/execute_notebooks.py`: three dependency-free notebooks pass
  and contain no committed outputs.
- `python3 scripts/audit_curriculum.py --strict`: 17/17 concept tracks satisfy
  substantive guide, metadata lesson, executable project/tests, separated
  solution, and scored stage assessment evidence.
- `python3 scripts/run_contract_tests.py`: 36/36 dependency-free suites pass
  with warnings treated as errors—17 curriculum projects and 19 academy
  contracts, totaling 82 tests. Its manifest is tested against the complete
  curriculum directory set.
- `python3 -m unittest discover -s tests -v`: twelve validator/runner tests pass.
- Detectron2 and MMDetection contract suites pass without compiled frameworks;
  FastAPI's four service integration tests pass in its declared environment.
- Fresh isolated environment `/tmp/handbook-core-env` on CPython 3.14 installed
  twelve lightweight academy requirement sets with `--no-cache-dir`; all 37
  native project tests pass with warnings treated as errors. This clean run
  found and fixed a Seaborn/Matplotlib pending-deprecation boundary before the
  final rerun. The environment is disposable and is not release source.
- Fresh isolated environment `/tmp/handbook-jax-env` on CPython 3.14 installed
  the declared CPU JAX, Keras, and Flax requirements with no cache reuse. All
  six native tests pass with warnings treated as errors, covering finite-
  difference gradients, seeded training, native Keras serialization, and Flax
  state round trips.
- `git diff --check` passes.

## Verification provided by repository automation

- `quality.yml` covers metadata, graph, core unit tests, dependency-free
  notebooks, curriculum project tests, and lightweight academy tests on Python
  3.12.
- `extended-academies.yml` isolates visualization, Dask, Gymnasium, NetworkX,
  OpenCV, and Albumentations environments.
- `nlp-academies.yml` isolates NLTK, spaCy, and Gensim.
- `heavy-academies.yml` and `compatibility-academies.yml` define scheduled/manual
  framework and version checks.
- `domain-framework-academies.yml` isolates ten pretrained NLP, generative,
  graph, and reinforcement-learning environments in separate Python 3.12 jobs.

Workflow definitions are not proof that a remote run succeeded. Record the
commit and run URL here before marking a release stable.

## Still required for a stable world-class release

- Execute every declared academy environment from a clean Python 3.12 runner,
  including DGL, PyTorch Geometric, Transformers, Diffusers, TensorFlow,
  PyTorch, JAX, detection frameworks, RL frameworks, and serving containers.
- Run real compiled-framework/model integration smoke tests where current labs
  deliberately validate contracts without downloading large artifacts.
- Record technical, pedagogical, accessibility, licensing, security, and
  responsible-use reviews; resolve high-severity findings.
- Conduct the required foundation and practitioner learner journeys and record
  blocking feedback plus remediation.
- Replace default repository-level ownership with additional independent domain
  reviewers as the contributor team grows.
- Expand clean-run evidence for paper reproduction, cross-framework benchmark,
  specialization deliverables, and the complete production capstone review.

Until those items are evidenced, metadata should remain `draft` or `review` and
the repository must not claim a versioned stable release.
