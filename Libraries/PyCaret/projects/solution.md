# PyCaret solution guidance

The reference uses the object-oriented experiment API, fixed session, bounded folds/threads, explicit target, and a strict inference schema. Its tiny dataset proves API wiring only. A real solution must inspect experiment configuration and transformed features, retain baselines, use deployment-shaped validation, avoid premature finalization, and verify the saved pipeline from a clean environment.
