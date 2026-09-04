"""Keras 3 smoke model using the CPU JAX backend."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-engineers-handbook-matplotlib")

import keras
import numpy as np


def data():
    x = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype="float32")
    return x, 2 * x + 1


def train(epochs=120):
    keras.utils.set_random_seed(7)
    model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(.1), loss="mse")
    x, y = data(); history = model.fit(x, y, epochs=epochs, verbose=0)
    return model, history.history["loss"]
