import functools

import jax
import jax.numpy as jnp


def nnext(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def adaptive_vmap(fn):
    @functools.wraps(fn)
    @jax.jit
    def f(*args):
        """

        Args:
            *args: (params, inputs) or (params, state, inputs)
                inputs: {u_net, y_net}
                    u_net: shape (Nu, Np) or (Np,)
                    y_net: shape (Nu, Ny, Ndim) or (Ny, Ndim) or (Ndim,)

        Returns:

        """
        inputs = args[-1]
        u_inputs = inputs["u_net"]
        # 1 -> (Np,) no vmap
        # 2 -> (Nu, Np) vmap over axis=0
        y_inputs = inputs["y_net"]
        y_in_axes = iter([0] * (y_inputs.ndim - 1))
        # 1 -> (Ncoords,) no vmap
        # 2 -> (Nsamples, Ncoords) vmap over axis=0
        # 3 -> (Nu, Nsamples, Ncoords) joint vmap over axis=0, then vmap over axis=1
        # u first, then y
        vfn = fn
        in_axes_base = (None,) * (len(args) - 1)
        if y_inputs.ndim > 1:
            in_axes = in_axes_base + ({"u_net": None, "y_net": nnext(y_in_axes)},)
            vfn = jax.vmap(vfn, in_axes=in_axes)
        if u_inputs.ndim == 2:
            in_axes = in_axes_base + ({"u_net": 0, "y_net": nnext(y_in_axes)},)
            vfn = jax.vmap(vfn, in_axes=in_axes)
        return vfn(*args)

    return f


def adaptive_vmap_p(fn):
    @functools.wraps(fn)
    @jax.jit
    def f(*args):
        """

        Args:
            *args: (params, parameters, inputs) or (params, state, parameters, inputs)
                parameters: shape (Nu, 1) or ()
                inputs: {u_net, y_net}
                    u_net: shape (Nu, Np) or (Np,)
                    y_net: shape (Nu, Ny, Ndim) or (Ny, Ndim) or (Ndim,)

        Returns:

        """
        parameters = args[-2]
        inputs = args[-1]
        u_inputs = inputs["u_net"]
        # 1 -> (Np,) no vmap
        # 2 -> (Nu, Np) vmap over axis=0
        y_inputs = inputs["y_net"]
        y_in_axes = iter([0] * (y_inputs.ndim - 1))
        # 1 -> (Ncoords,) no vmap
        # 2 -> (Nsamples, Ncoords) vmap over axis=0
        # 3 -> (Nu, Nsamples, Ncoords) joint vmap over axis=0, then vmap over axis=1
        # u first, then y
        vfn = fn
        in_axes_base = (None,) * (len(args) - 2)
        if y_inputs.ndim > 1:
            in_axes = (
                in_axes_base + (None,) + ({"u_net": None, "y_net": nnext(y_in_axes)},)
            )

            vfn = jax.vmap(vfn, in_axes=in_axes)
        if u_inputs.ndim == 2:
            in_axes = in_axes_base + (0,) + ({"u_net": 0, "y_net": nnext(y_in_axes)},)
            vfn = jax.vmap(vfn, in_axes=in_axes)
            # parameters: values shape (Nu,1) -> (Nu,)
            parameters = {
                k: jnp.reshape(v, newshape=(v.shape[0])) for k, v in parameters.items()
            }
        return vfn(*args[:-2], parameters, inputs)

    return f


def adaptive_vmap_s(fn):
    @functools.wraps(fn)
    @jax.jit
    def f(*args):
        """

        Args:
            *args: (params, inputs) or (params, state, inputs)
                data:
                    inputs: {u_net, y_net}
                        u_net: shape (Nu, Np) or (Np,)
                        y_net: shape (Nu, Ny, Ndim) or (Ny, Ndim) or (Ndim,)
                    s: shape (Nu, Ny, Nout), (Nu, Ny) (Ny,) or ()

        Returns:

        """
        inputs = args[-1]["inputs"]
        s = args[-1]["s"]
        # s.shape (Nu, Ny) or ()
        s_in_axes = iter([0] * s.ndim)

        u_inputs = inputs["u_net"]
        # 1 -> (Np,) no vmap
        # 2 -> (Nu, Np) vmap over axis=0
        y_inputs = inputs["y_net"]
        y_in_axes = iter([0] * (y_inputs.ndim - 1))
        # 1 -> (Ncoords,) no vmap
        # 2 -> (Nsamples, Ncoords) vmap over axis=0
        # 3 -> (Nu, Nsamples, Ncoords) joint vmap over axis=0, then vmap over axis=1
        # u first, then y
        vfn = fn
        in_axes_base = (None,) * (len(args) - 1)
        if y_inputs.ndim > 1:
            in_axes = in_axes_base + (
                {
                    "inputs": {"u_net": None, "y_net": nnext(y_in_axes)},
                    "s": nnext(s_in_axes),
                },
            )
            vfn = jax.vmap(vfn, in_axes=in_axes)
        if u_inputs.ndim == 2:
            in_axes = in_axes_base + (
                {
                    "inputs": {"u_net": 0, "y_net": nnext(y_in_axes)},
                    "s": nnext(s_in_axes),
                },
            )
            vfn = jax.vmap(vfn, in_axes=in_axes)

        return vfn(*args)

    return f


def adaptive_vmap_p_s(fn):
    @functools.wraps(fn)
    @jax.jit
    def f(*args):
        """

        Args:
            *args: (params, parameters, inputs) or (params, state, parameters, inputs)
                parameters: shape (Nu, 1) or ()
                data:
                    inputs: {u_net, y_net}
                        u_net: shape (Nu, Np) or (Np,)
                        y_net: shape (Nu, Ny, Ndim) or (Ny, Ndim) or (Ndim,)
                    s: shape (Nu, Ny, Nout), (Nu, Ny) (Ny,) or ()

        Returns:

        """
        parameters = args[-2]
        inputs = args[-1]["inputs"]
        s = args[-1]["s"]
        s_in_axes = iter([0] * s.ndim)

        u_inputs = inputs["u_net"]
        # 1 -> (Np,) no vmap
        # 2 -> (Nu, Np) vmap over axis=0
        y_inputs = inputs["y_net"]
        y_in_axes = iter([0] * (y_inputs.ndim - 1))
        # 1 -> (Ncoords,) no vmap
        # 2 -> (Nsamples, Ncoords) vmap over axis=0
        # 3 -> (Nu, Nsamples, Ncoords) joint vmap over axis=0, then vmap over axis=1
        # u first, then y
        vfn = fn
        in_axes_base = (None,) * (len(args) - 2)
        if y_inputs.ndim > 1:
            in_axes = (
                in_axes_base
                + (None,)
                + (
                    {
                        "inputs": {"u_net": None, "y_net": nnext(y_in_axes)},
                        "s": nnext(s_in_axes),
                    },
                )
            )
            vfn = jax.vmap(vfn, in_axes=in_axes)
        if u_inputs.ndim == 2:
            in_axes = (
                in_axes_base
                + (0,)
                + (
                    {
                        "inputs": {"u_net": 0, "y_net": nnext(y_in_axes)},
                        "s": nnext(s_in_axes),
                    },
                )
            )
            vfn = jax.vmap(vfn, in_axes=in_axes)
            # parameters: values shape (Nu,1) -> (Nu,)
            parameters = {
                k: jnp.reshape(v, newshape=(v.shape[0])) for k, v in parameters.items()
            }

        return vfn(*args[:-2], parameters, args[-1])

    return f
