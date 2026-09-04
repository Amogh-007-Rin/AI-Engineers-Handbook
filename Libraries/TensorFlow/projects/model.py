"""Deterministic TensorFlow regression and SavedModel fixture."""

import tensorflow as tf


def data():
    x = tf.constant([[-1.0], [0.0], [1.0], [2.0]], tf.float32)
    return x, 2.0 * x + 1.0


class Regressor(tf.Module):
    def __init__(self):
        super().__init__()
        self.weight = tf.Variable([[0.0]], name="weight")
        self.bias = tf.Variable([0.0], name="bias")

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="features")])
    def serve(self, features):
        return {"predictions": features @ self.weight + self.bias}


def train(steps=120):
    tf.keras.utils.set_random_seed(7)
    model, optimizer = Regressor(), tf.keras.optimizers.SGD(0.1)
    x, y = data()
    losses = []
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model.serve(x)["predictions"] - y))
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(g is None for g in gradients):
            raise RuntimeError("missing gradient")
        tf.debugging.assert_all_finite(gradients, "nonfinite gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(float(loss))
    return model, losses
