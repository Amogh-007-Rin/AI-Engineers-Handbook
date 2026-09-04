# Library Academies

Library academies teach individual tools from first contact through production
practice. They complement the [concept-first curriculum](../curriculum/README.md):
learn the underlying idea in the curriculum, then use an academy to understand
one tool's abstractions, failure modes, and operational tradeoffs.

## Coverage

All 60 academies have a substantive learning slice containing a tailored guide,
metadata lesson, exercises, executable project and tests, separated solution,
scored assessment, and declared environment. This is automated structural and
execution evidence—not a substitute for the independent reviews and learner
journeys required for “complete academy” status.

| Family | Count | Examples |
|---|---:|---|
| Data and scientific computing | 8 | NumPy, Pandas, SciPy, Polars, DuckDB, Dask, PySpark |
| Visualization | 5 | Matplotlib, Seaborn, Plotly, Bokeh, Dash |
| Classical ML and forecasting | 9 | Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna, Prophet |
| Deep learning and interchange | 7 | PyTorch, TensorFlow, JAX, Keras, Flax, FastAI, ONNX |
| NLP and generative AI | 8 | Transformers, Diffusers, spaCy, NLTK, Rasa, OpenAI SDK |
| Computer vision | 6 | OpenCV, Albumentations, YOLO, Detectron2, MMDetection |
| Reinforcement learning | 4 | Gymnasium, PettingZoo, Stable-Baselines3, RLlib |
| MLOps and AI systems | 10 | MLflow, W&B, Airflow, BentoML, FastAPI, Docker, Kubernetes |
| Graphs and graph learning | 3 | NetworkX, PyTorch Geometric, DGL |

Browse every entry in the [categorized academy catalog](CATALOG.md), and use the
[primary source registry](SOURCES.md) for official documentation, source
repositories, and research starting points.

## Completion levels

- **Foundation:** Install the tool, explain its mental model, and complete
  essential operations safely.
- **Practitioner:** Build, test, debug, and evaluate an end-to-end workflow.
- **Advanced:** Optimize, extend, integrate, and operate the tool while
  defending tradeoffs against credible alternatives.
- **Maintainer-ready:** Read internals, diagnose deeper failures, design an
  upstream-quality change, and understand compatibility/governance constraints.

## Recommended workflow

1. Read the academy README and record the declared version/runtime.
2. Create an isolated environment from `environment/requirements.txt`.
3. Work through the fundamentals lesson and predict behavior before execution.
4. Complete exercises without reading the solution first.
5. Run the project tests with warnings treated as errors where supported.
6. Complete the assessment and defend when the tool should *not* be used.
7. Record compatibility, security, licensing, performance, and operational
   findings in your project evidence.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Libraries/NumPy/environment/requirements.txt
python -W error -m unittest discover -s Libraries/NumPy/projects -v
```

Do not install all academies into one environment. Framework ecosystems have
conflicting runtime and binary constraints; the CI workflows deliberately test
them in isolated jobs.

## Maturity and evidence

Use [the verification ledger](../reports/verification-status.md) to distinguish:

- dependency-free contract evidence;
- clean native environment evidence;
- Python/runtime compatibility boundaries;
- workflow definitions that have not yet produced a recorded remote run; and
- independent review or learner evidence that still requires people.

An academy is not “published” merely because its files exist or tests pass.
Publication follows the definition of done in [project.md](../project.md) and
the cadence in [MAINTENANCE.md](../MAINTENANCE.md).
