# Solution notes

Recursive redaction happens before the tracking SDK receives configuration.
The native fixture proves that the sanitized config, metric, and artifact enter
a real offline run while caches remain inside an owned directory. Production
code should also minimize environment capture, encrypt retained runs, and
attach data/code hashes to every run and artifact.
