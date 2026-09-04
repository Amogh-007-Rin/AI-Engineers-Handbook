"""Offline Transformers configuration and batch-contract fixture."""

from transformers import AutoConfig


def config(num_labels=3):
    if num_labels < 2:
        raise ValueError("classification requires at least two labels")
    return AutoConfig.for_model("bert", vocab_size=128, hidden_size=32,
                                num_hidden_layers=1, num_attention_heads=4,
                                intermediate_size=64, num_labels=num_labels)


def validate_batch(batch, *, max_length=16):
    required = ("input_ids", "attention_mask")
    if any(key not in batch for key in required):
        raise ValueError("input_ids and attention_mask required")
    if len(batch["input_ids"]) != len(batch["attention_mask"]):
        raise ValueError("batch sizes differ")
    for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
        if len(ids) != len(mask) or len(ids) > max_length or any(bit not in (0, 1) for bit in mask):
            raise ValueError("invalid token/mask contract")
    return True
