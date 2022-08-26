import functools

import jax
import jax.numpy as jnp
import pytest

import onet_disk2D.model
import onet_disk2D.vmap
from onet_disk2D.vmap import nnext


def s_fn(params, state, inputs):
    """

    Args:
        params:
        state:
        inputs:
            u_net: (..., 4)
            y_net: (..., 2)

    Returns:

    """
    u = inputs["u_net"]
    if u.ndim > 1:
        u = jnp.reshape(u, (u.shape[0], 1, u.shape[-1]))
    y = inputs["y_net"]
    out = u[..., 0] + y[..., 0]
    return out


def s_fn_p(params, state, parameters, inputs):
    out = s_fn(params, state, inputs)
    return parameters["a"] * out


def s_fn_s(params, state, data):
    inputs = data["inputs"]
    s = data["s"]
    out = s_fn(params, state, inputs)
    return out - s


def s_fn_p_s(params, state, parameters, data):
    inputs = data["inputs"]
    s = data["s"]
    out = s_fn_p(params, state, parameters, inputs)
    return out - s


def adaptive_vmap2(fn):
    @functools.wraps(fn)
    @jax.jit
    def f(*args):
        inputs = args[-1]
        u_inputs = inputs["u2_net"]
        # 1 -> (Np,) no vmap
        # 2 -> (Nu, Np) vmap over axis=0
        y_inputs = inputs["y2_net"]
        y_in_axes = iter([0] * (y_inputs.ndim - 1))
        # 1 -> (Ncoords,) no vmap
        # 2 -> (Nsamples, Ncoords) vmap over axis=0
        # 3 -> (Nu, Nsamples, Ncoords) joint vmap over axis=0, then vmap over axis=1
        # u first, then y
        vfn = fn
        in_axes_base = (None,) * (len(args) - 1)
        if y_inputs.ndim > 1:
            in_axes = in_axes_base + ({"u2_net": None, "y2_net": nnext(y_in_axes)},)
            vfn = jax.vmap(vfn, in_axes=in_axes)
        if u_inputs.ndim == 2:
            in_axes = in_axes_base + ({"u2_net": 0, "y2_net": nnext(y_in_axes)},)
            vfn = jax.vmap(vfn, in_axes=in_axes)
        return vfn(*args)

    return f


u = jnp.linspace(0.0, 1.0, 12).reshape((3, 4))

r_min = 0.4
r_max = 2.5
Nr = 5
Ntheta = 3
r = jnp.linspace(r_min, r_max, Nr)
theta_min = -jnp.pi
theta_max = jnp.pi
theta = jnp.linspace(theta_min, theta_max, Ntheta)

y = jnp.stack(jnp.meshgrid(r, theta, indexing="ij"), axis=-1)
y = y.reshape((-1, 2))

multi_y = jnp.broadcast_to(y, (u.shape[0],) + y.shape)

parameters = {"a": jnp.linspace(0.0, 1.0, 3)[..., None]}

s1 = 0.0
s2 = jnp.zeros((3, 15))


@pytest.fixture(params=[y, multi_y])
def inputs(request):
    return {"u_net": u, "y_net": request.param}


@pytest.fixture(params=[s1, s2])
def s(request):
    return request.param


@pytest.fixture
def data(inputs, s):
    return {"inputs": inputs, "s": s}


def test_inputs(inputs):
    cri = [
        inputs["u_net"].shape == (3, 4),
        inputs["y_net"].shape[-1] == 2,
    ]
    assert all(cri)


# parameterize: s_fn, args, vmap function
# test:
#  direct apply output shape
#  vmap apply output shape
#  ds apply output shape
#  dds apply output shape


def test_direct_apply(inputs, data):
    s = s_fn(None, None, inputs)
    s_p = s_fn_p(None, None, parameters, inputs)
    s_s = s_fn_s(None, None, data)
    s_p_s = s_fn_p_s(None, None, parameters, data)
    cri = [
        s.shape == (3, 15),
        s_p.shape == (3, 15),
        s_s.shape == (3, 15),
        s_p_s.shape == (3, 15),
    ]
    assert all(cri)


def test_vmap_apply(inputs, data):
    s = onet_disk2D.vmap.adaptive_vmap(s_fn)(None, None, inputs)
    s_p = onet_disk2D.vmap.adaptive_vmap_p(s_fn_p)(None, None, parameters, inputs)
    s_s = onet_disk2D.vmap.adaptive_vmap_s(s_fn_s)(None, None, data)
    s_p_s = onet_disk2D.vmap.adaptive_vmap_p_s(s_fn_p_s)(None, None, parameters, data)
    cri = [
        s.shape == (3, 15),
        s_p.shape == (3, 15),
        s_s.shape == (3, 15),
        s_p_s.shape == (3, 15),
    ]
    assert all(cri)


def test_dsdy_apply(inputs, data):
    ds = onet_disk2D.vmap.adaptive_vmap(jax.jacfwd(s_fn, argnums=-1))(
        None, None, inputs
    )
    dsdy = ds["y_net"]
    ds_p = onet_disk2D.vmap.adaptive_vmap_p(jax.jacfwd(s_fn_p, argnums=-1))(
        None, None, parameters, inputs
    )
    dsdy_p = ds_p["y_net"]
    cir = [
        dsdy.shape == (3, 15, 2),
        dsdy_p.shape == (3, 15, 2),
    ]
    assert all(cir)
