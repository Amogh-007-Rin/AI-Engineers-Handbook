# Solution notes

`AutoConfig.for_model` exercises the library registry without downloading
weights. The batch validator exposes padding/mask assumptions. A full solution
pins model/tokenizer revisions and proves offline artifact parity.
