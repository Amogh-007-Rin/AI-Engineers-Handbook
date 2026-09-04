"""Pure JAX linear training fixture with explicit parameters."""

import jax
import jax.numpy as jnp


def data():
    x = jnp.array([[-1.0], [0.0], [1.0], [2.0]])
    return x, 2 * x + 1


def loss(params, x, y):
    prediction = x * params["weight"] + params["bias"]
    return jnp.mean((prediction - y) ** 2)


@jax.jit
def step(params, x, y, learning_rate):
    value, gradients = jax.value_and_grad(loss)(params, x, y)
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, gradients), value


def train(steps=200):
    params = {"weight": jnp.array(0.0), "bias": jnp.array(0.0)}
    x, y = data()
    for _ in range(steps): params, value = step(params, x, y, .1)
    return params, float(value)
