# Solution guidance

Make state, action, reward, horizon, termination, truncation, and seeds explicit.
Compare random and heuristic policies before learning. Evaluate return/success
distributions across seeds and shifts. Separate exploration from evaluation,
checkpoint normalization/replay state, and test invalid actions, safety limits,
timeouts, and clean restore.
