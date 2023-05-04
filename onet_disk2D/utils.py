import time

import chex
import jax
import jax.numpy as jnp


@jax.jit
def to_log(u: chex.Array, col_idx_to_apply: chex.Array) -> chex.Array:
    """Convert the second axis of u to log scale.

    Args:
        u: shape (n_samples, n_features)
        col_idx_to_apply: jax boolean array of which column to apply the transformation. E.g. [True, False, True] for the first and third columns.

    Returns:

    """

    def apply_log_transform(col, apply_log):
        return jnp.where(apply_log, jnp.log10(col), col)

    u = jax.vmap(apply_log_transform, in_axes=(-1, 0), out_axes=-1)(u, col_idx_to_apply)

    return u


@jax.jit
def to_linear(u: chex.Array, col_idx_to_apply: chex.Array) -> chex.Array:
    """Convert the columns of u to linear scale.

    Args:
        u:
        col_idx_to_apply: jax boolean array of which column to apply the transformation. E.g. [True, False, True] for the first and third columns.

    """

    def apply_log_transform(col, apply_log):
        return jnp.where(apply_log, 10.0**col, col)

    u = jax.vmap(apply_log_transform, in_axes=(-1, 0), out_axes=-1)(u, col_idx_to_apply)

    return u


# timer function decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} time: {time.perf_counter() - start_time:.2f} seconds")
        return result

    return wrapper
