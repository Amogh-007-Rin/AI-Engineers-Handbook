# Solution notes

Parallel steps require aligned per-agent mappings; mismatched keys otherwise
produce subtle reward attribution bugs. A full environment must additionally
validate spaces, action masks, dead-agent steps, and seed determinism.
