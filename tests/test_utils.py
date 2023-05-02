import jax.numpy as jnp

import onet_disk2D.utils


def test_to_log_1():
    u = jnp.array([1.0, 10.0, 100.0])
    col_idx_to_apply = jnp.array([True, False, True])
    u = onet_disk2D.utils.to_log(u, col_idx_to_apply)
    assert jnp.allclose(u, jnp.array([0.0, 10.0, 2.0]))


def test_to_log_2():
    u = jnp.array([1.0, 10.0, 100.0])
    u = jnp.stack([u, u, u], axis=0)

    col_idx_to_apply = jnp.array([True, False, True])
    log_u = jnp.array([0.0, 10.0, 2.0])
    log_u = jnp.stack([log_u, log_u, log_u], axis=0)

    u = onet_disk2D.utils.to_log(u, col_idx_to_apply)
    assert jnp.allclose(u, log_u)


def test_to_linear_1():
    u = jnp.array([0.0, 10.0, 2.0])
    col_idx_to_apply = jnp.array([True, False, True])
    u = onet_disk2D.utils.to_linear(u, col_idx_to_apply)
    assert jnp.allclose(u, jnp.array([1.0, 10.0, 100.0]))


def test_to_linear_2():
    log_u = jnp.array([0.0, 10.0, 2.0])
    log_u = jnp.stack([log_u, log_u, log_u], axis=0)

    col_idx_to_apply = jnp.array([True, False, True])
    u = jnp.array([1.0, 10.0, 100.0])
    u = jnp.stack([u, u, u], axis=0)

    log_u = onet_disk2D.utils.to_linear(log_u, col_idx_to_apply)
    assert jnp.allclose(log_u, u)
