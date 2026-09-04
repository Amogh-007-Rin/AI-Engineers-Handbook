# Solution notes

Fixed validation indices make split assertions possible. In a real task choose
group- or time-aware indices first. Add seeded repetitions, baseline and slice
metrics, then perform artifact reload in a clean subprocess.

`Learner.export`/`load_learner` uses pickle. Treat the file as executable code:
accept it only from a trusted build, verify a cryptographic digest or signature,
restrict read access, and never deserialize learner files uploaded by users.
