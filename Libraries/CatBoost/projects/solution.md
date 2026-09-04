# CatBoost solution guidance

The reference bounds symmetric-tree depth, rounds, threads, randomness, and file side effects; checks held-out ranking and native persistence. The extension must explicitly name categorical features, test unseen/missing category strings, compare with a leakage-safe one-hot baseline, and preserve the full feature contract alongside the model.
