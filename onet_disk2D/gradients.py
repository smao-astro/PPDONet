from typing import Iterable, Hashable

import chex
import jax
import jax.numpy as jnp


@jax.jit
def sum_gradients(g: Iterable[chex.ArrayTree]) -> chex.ArrayTree:
    return jax.tree_map(lambda *args: jnp.sum(jnp.asarray(args), axis=0), *g)


@jax.jit
def sum_weighted_gradients(
    g: dict[Hashable, chex.ArrayTree], w: dict[Hashable, float]
) -> chex.ArrayTree:
    g = [jax.tree_map(lambda x: w[k] * x, _g) for k, _g in g.items()]
    return sum_gradients(g)
