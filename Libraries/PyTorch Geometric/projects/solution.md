# Solution notes

The explicit expected edge index makes PyG's batching increment observable.
Production systems must preserve original entity IDs and graph/split schema
alongside checkpoints so reindexed tensors cannot silently change meaning.
