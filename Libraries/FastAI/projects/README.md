# fastai tabular contract project

This CPU lab makes the hidden pipeline inspectable: a fixed disjoint split,
train-only processors, a small learner, and an export/reload prediction. Run
under the reference Python 3.12 environment with `python -W error -m unittest
-v`. The test explicitly asserts FastAI's pickle warning: exported learners
are executable Python artifacts and must never be loaded from an untrusted
source. Extend the lab with a signed manifest and a clean inference process.
