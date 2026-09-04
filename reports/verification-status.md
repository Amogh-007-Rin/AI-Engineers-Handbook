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
- `python3 -m unittest discover -s tests -v`: fourteen validator/runner tests pass.
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
- Fresh isolated CPU-only PyTorch environment on CPython 3.14 installed from
  the official PyTorch CPU index. Both native tests pass warning-clean on
  PyTorch 2.14.0+cpu and NumPy 2.5.2, covering seeded training and state-dict
  serialization. The run exposed and fixed a missing NumPy environment
  dependency.
- TensorFlow CPU installation was attempted in a fresh CPython 3.14 environment
  and correctly failed with no compatible `tensorflow-cpu` distribution. Its
  declared reference and CI runtime remains CPython 3.12; this local result is
  compatibility-boundary evidence, not a passing TensorFlow execution.
- Fresh isolated CPython 3.14 boosting/forecasting environment installed the
  declared XGBoost, LightGBM, CatBoost, and Prophet requirements with no cache
  reuse. All eight native tests pass with warnings treated as errors, covering
  seeded reproducibility, training quality, native serialization, forecast
  horizons, and trend behavior. The run also identified and fixed Prophet's
  writable Matplotlib configuration boundary.
- Fresh isolated CPython 3.14 ONNX environment installed ONNX 1.22.0, ONNX
  Runtime 1.29.0, and NumPy 2.5.2 with no cache reuse. Both native differential
  inference/dynamic-batch and file-round-trip tests pass warning-clean at the
  Python level. ONNX Runtime logs a non-fatal native warning when it cannot
  persist its anonymous telemetry device ID under the managed read-only home;
  inference falls back to an in-memory identifier.
- Fresh isolated CPython 3.14 NLP environment installed NLTK 3.10.3 and spaCy
  3.8.16 with no cache reuse. All six native project tests pass with warnings
  treated as errors, covering deterministic vocabulary construction, malformed
  input, token contracts, character offsets, document ordering, and spaCy disk
  round trips.
- Gensim 4.4.0 installation was attempted in the same clean CPython 3.14
  environment. Its source distribution fails to compile because generated C
  code accesses CPython integer internals removed in 3.14. Gensim therefore
  remains assigned to the repository's clean Python 3.12 NLP job; this is a
  recorded compatibility boundary, not passing Gensim execution evidence.
- Fresh isolated CPython 3.14 extended-academy environment installed
  Albumentations 2.0.8, Bokeh 3.10.0, Dash 3.4.0, Dask 2025.12.0, Gymnasium
  1.3.0, NetworkX 3.6.1, OpenCV 4.14.0.94, and Plotly 6.9.0. All 17 native
  project tests pass with warnings treated as errors. The run exposed and
  fixed Albumentations' import-time network update check and a NetworkX 3.6
  node-link serialization-key drift; both contracts are now explicit and
  offline-safe.
- Fresh isolated CPython 3.14 CPU PyTorch domain environment installed
  Transformers 4.57.6, Diffusers 0.39.0, SentenceTransformers 3.4.1, PyTorch
  Geometric 2.8.0.post1, and Stable-Baselines3 2.9.0 over PyTorch 2.14.0+cpu.
  The three Hugging Face academies pass all six native tests warning-clean and
  offline without pretrained downloads. Stable-Baselines3 passes three tests,
  including a newly added real seeded PPO train/predict lifecycle rather than
  only a configuration mock.
- Both PyTorch Geometric graph batching tests pass functionally on CPython
  3.14. Its import emits PyTorch's documented `torch.jit.script` Python 3.14
  `FutureWarning`, so this is not warning-clean evidence; the strict academy
  reference remains the isolated Python 3.12 domain-framework workflow.
- FastAI 2.8.8 installs in the clean CPython 3.14 CPU environment and both
  native tabular split and export/reload tests pass functionally. Importing its
  dependency chain exposes CPython 3.14 deprecations and PyTorch's JIT warning,
  so warning-strict evidence remains assigned to Python 3.12. The run also
  strengthened the test and lesson to assert and explain the untrusted-pickle
  boundary of `load_learner` rather than suppressing its security warning.
- Fresh isolated CPython 3.14 SDK/MLOps environments installed OpenAI Python
  1.109.1, MLflow 3.16.0, and Weights & Biases 0.29.0. Each academy passes
  three native tests with warnings treated as errors. The OpenAI test performs
  a real typed Responses API request/response round trip through an injected
  offline transport with no key or billable call. MLflow creates, terminates,
  and reads a real SQLite-backed run with provenance, metrics, and a signature
  artifact. W&B uses real offline mode to log a recursively redacted config,
  metric, and model-card artifact with state scoped to a disposable directory.
- The MLflow run identified an obsolete `<3` academy cap and the maintenance-
  only legacy file tracking backend. The environment now targets MLflow 3.16+
  and the project uses SQLite tracking with an explicit artifact location.
- BentoML 1.4.39 installs on CPython 3.14 and all five project tests pass
  functionally, including construction and in-process execution of a real
  typed service with explicit traffic limits. Its pinned `cattrs`/`pathspec`
  dependency chain emits Python 3.14 deprecations, so BentoML remains a
  warning-strict Python 3.12 CI target rather than hiding upstream warnings.
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
- `mlops-academies.yml` runs real offline OpenAI SDK, MLflow, and W&B lifecycle
  tests independently on Python 3.12 and 3.14, plus BentoML on its strict
  Python 3.12 reference runtime.

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
