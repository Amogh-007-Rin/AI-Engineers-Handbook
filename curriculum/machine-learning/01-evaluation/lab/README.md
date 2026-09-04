# Binary evaluation lab

This offline lab keeps counts, ratios, costs, sampling variability, and slices
visible. Run from the repository root:

```bash
python3 curriculum/machine-learning/01-evaluation/lab/evaluation.py
python3 -m unittest discover -s curriculum/machine-learning/01-evaluation/lab -v
```

For the fixture, threshold `0.6` has zero declared error cost, all four labels
are classified correctly, and slice support sums to four. Eight tests pass.

## Investigation

Predict results at thresholds `0.4`, `0.6`, and `0.8`, then change false-positive
and false-negative costs independently. Create a one-class slice and explain why
its precision or recall can be undefined. Repeat the bootstrap with several
seeds and distinguish Monte Carlo variation from deployment uncertainty.

## Limitations

The percentile bootstrap resamples rows as independent units and is educational,
not a universal interval method. The lab does not compute ranking curves, causal
effects, fairness conclusions, label quality, or deployment shift. Its threshold
must be chosen on validation evidence; using test labels would invalidate the
protocol even though the function returns a correct minimum.
