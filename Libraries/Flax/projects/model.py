"""Minimal Flax Linen model trained with explicit JAX parameters."""

import flax.linen as nn
from functools import partial
import jax
import jax.numpy as jnp


class Linear(nn.Module):
    @nn.compact
    def __call__(self, x): return nn.Dense(1)(x)


def data():
    x = jnp.array([[-1.0], [0.0], [1.0], [2.0]]); return x, 2*x + 1


def loss(params, model, x, y): return jnp.mean((model.apply({"params": params}, x) - y) ** 2)


@partial(jax.jit, static_argnames=("model",))
def step(params, model, x, y):
    value, grads = jax.value_and_grad(loss)(params, model, x, y)
    return jax.tree.map(lambda p, g: p - .1*g, params, grads), value


def train(steps=200):
    model = Linear(); x, y = data()
    params = model.init(jax.random.key(7), x)["params"]
    for _ in range(steps): params, value = step(params, model, x, y)
    return model, params, float(value)
