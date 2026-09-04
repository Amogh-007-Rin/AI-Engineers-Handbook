# Transformers offline contract

Create a tiny BERT configuration without network access and validate token/mask
batches. Run `python -W error -m unittest -v`; extend with a local tokenizer,
tiny random model, training step, artifact save/reload, and slice evaluation.
