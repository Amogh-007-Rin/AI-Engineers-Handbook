# Solution notes

The fixture rejects ambiguous edge semantics at construction and sorts node
keys for deterministic output. Production work should version the node schema,
protect identifiers, and choose splits that prevent neighborhood leakage.
