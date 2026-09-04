# Keras solution guidance

The reference selects its backend before importing Keras, sets a shared seed, declares input shape, uses a bounded explicit optimizer, verifies loss, and tests native-format reload. A full solution restores the best validation checkpoint, tests training/inference behavior, uses `keras.ops` in portable layers, and reruns from clean backend-specific environments.
