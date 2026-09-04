# RLlib experiment contract

Validate algorithm, environment, runner resources, evaluation, and checkpoint
policy without starting a cluster. Run `python -W error -m unittest -v`; extend
with a local PPO run, clean checkpoint restore, scaling comparison, and injected
worker failure.
