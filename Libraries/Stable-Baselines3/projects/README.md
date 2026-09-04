# Stable-Baselines3 experiment contract

Validate algorithm/action-space compatibility, seed count, evaluation budget,
and frozen normalization before consuming compute. In the declared environment,
the native test also constructs, trains, and queries a seeded CPU PPO agent on
`CartPole-v1`. Its 32 timesteps prove framework and environment integration,
not task mastery; a result still needs the multi-seed evaluation contract.

Run `python -W error -m unittest -v`. Without the optional dependency, the pure
contract checks run and the native test is visibly skipped rather than mocked.
Extend it with environment checking, artifact reload, and shifted evaluation.
