# Solution notes

The dictionary owns the feature-space identity and must travel with the model.
The OOV test catches a common leakage bug: updating vocabulary during
validation. A full submission adds streaming corpus fixtures, filtering
thresholds, multiple seeds for latent models, and task-based evaluation.
