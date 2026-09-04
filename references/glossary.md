# Core glossary

Terms are intentionally operational: each definition should help make a design
or evaluation decision.

- **Ablation:** controlled removal or alteration of one component to estimate its contribution while holding other conditions fixed.
- **Baseline:** the simplest credible comparator, including a non-ML rule or existing process when relevant.
- **Calibration:** agreement between predicted confidence and observed outcome frequency for a defined population and period.
- **Concept drift:** change in the relationship between inputs and targets; distinct from a change in input distribution alone.
- **Data leakage:** information available during development that would not be legitimately available at prediction time or across the intended split.
- **Data lineage:** traceable history of data sources, transformations, versions, and consumers.
- **Determinism:** repeated execution under declared conditions produces the same result; deterministic code can still be scientifically invalid.
- **Distribution shift:** deployment data differ from development data. Name which distribution changed and how it was measured.
- **Embedding:** learned or designed vector representation whose geometry is useful only relative to a task, model, and similarity definition.
- **Grounding:** constraining or evaluating a generated response against an identified source or environment; retrieval alone does not prove grounding.
- **Idempotency:** safely repeating an operation has the same externally visible effect as performing it once.
- **Inference:** model execution to produce predictions; in statistics it may instead mean reasoning about a population from observed data.
- **Latency percentile:** duration below which a stated percentage of requests complete, reported with workload, window, and measurement boundary.
- **Model card:** scoped report of intended use, evaluation, limitations, risks, and maintenance—not a marketing page.
- **Observability:** ability to infer system state from traces, metrics, logs, and artifacts tied to actionable questions.
- **Precision / recall:** respectively, correct positive predictions among predicted positives and recovered positives among actual positives. Always state the positive class and threshold.
- **Provenance:** evidence of origin, custody, license, and transformation for data, models, code, and claims.
- **Reproducibility:** another execution recreates a result from declared code, data, environment, configuration, and procedure within stated tolerance.
- **Rollback:** tested restoration to a known acceptable system and state; it includes schema and data compatibility, not only replacing a model file.
- **Seed:** initial state for a pseudorandom process. One seed aids debugging; multiple seeds help estimate instability.
- **SLO:** measurable reliability target for a service level indicator over a defined window, with an error budget and response policy.
- **Threat model:** structured account of assets, actors, trust boundaries, attack paths, controls, and residual risks.
- **Uncertainty:** lack of certainty about data, parameters, predictions, or consequences. Report the type and estimation assumptions.

When a term has discipline-specific meanings, define the meaning used in the
artifact rather than assuming a universal interpretation.
