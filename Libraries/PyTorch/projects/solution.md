# PyTorch solution guidance

The reference uses explicit gradient clearing, finite-loss checks, seeded initialization, state-dict-only persistence, `weights_only=True`, and inference without gradient recording. The tiny linear task proves loop wiring rather than generalization. Extensions must separate splits, restore best validation state, test train/eval behavior, and report synchronization-aware timing.
