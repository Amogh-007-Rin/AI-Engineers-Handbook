# Vectorized feature engine solution notes

The reference implementation keeps validation at the array boundary, stores learned statistics with a retained feature axis, replaces zero scales with one, and never mutates caller input. Pairwise interactions use upper-triangle feature indices to avoid duplicate/self products. Cosine similarity assigns zero similarity to zero-norm rows rather than silently dividing by zero.

Run:

```bash
python3 -m unittest -v test_vectorized_features.py
```

Reviewers should additionally ask learners to compare outputs with a loop oracle, explain interaction-column ordering, measure allocation for a larger similarity matrix, and propose chunking when the quadratic output cannot fit memory.
