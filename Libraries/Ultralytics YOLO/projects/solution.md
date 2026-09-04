# Solution notes

The validator checks both normalized centers and box edges; center-only checks
miss annotations extending beyond the image. Production work also detects
duplicate frames, class drift, corrupt images, and licensing conflicts.
