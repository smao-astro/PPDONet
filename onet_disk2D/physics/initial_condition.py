import abc
import functools
import typing

import jax
import jax.numpy as jnp

import onet_disk2D.physics.pde


class ICDict(typing.TypedDict):
    ic_sigma: typing.Any
    ic_v_r: typing.Any
    ic_v_theta: typing.Any


class IC:
    def __init__(self, index, **kwargs):
        self.index = index

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """

        return f

    def res_fn(self, s_fn):
        """The passing of s_fn is deferred to this call so that IC can be used in build_model."""

        @jax.jit
        def f(params, state, parameters, inputs, s):
            # the shape for the terms below: (N, 1)
            u = s_fn(params, state, parameters, inputs)[..., self.index]
            return u - self.func(parameters, inputs["y_net"]) - s

        return f

    def loss_fn(self, s_fn):
        """The passing of s_fn is deferred to this call so that IC can be used in build_model."""
        res_fn = self.res_fn(s_fn)

        @jax.jit
        def f(*args):
            return jnp.mean(res_fn(*args) ** 2)

        return f


class RingSigmaIC(IC):
    def __init__(self, **kwargs):
        super(RingSigmaIC, self).__init__(index=0, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            ic_values = parameters["sigma0"] * (
                1.0
                + jnp.exp(
                    -((r - parameters["ringcenter"]) ** 2)
                    / (2 * parameters["ringwidth"] ** 2)
                )
                / (jnp.sqrt(2 * jnp.pi) * parameters["ringwidth"])
            )
            return ic_values

        return f


class PowerlawSigmaIC(IC):
    def __init__(self, **kwargs):
        super(PowerlawSigmaIC, self).__init__(index=0, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            ic_values = parameters["sigma0"] * r ** -parameters["sigmaslope"]
            return ic_values

        return f


class KeplerianVThetaIC(IC):
    def __init__(self, omegaframe, **kwargs):
        self.omegaframe = omegaframe
        super(KeplerianVThetaIC, self).__init__(index=2, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            out = r**-0.5 - self.omegaframe * r
            if parameters["aspectratio"].ndim > 1:
                return jnp.broadcast_to(
                    out, (parameters["aspectratio"].shape[0], r.shape[-1])
                )
            else:
                return out

        return f


class StaticPowerlawVThetaIC(IC):
    def __init__(self, omegaframe, **kwargs):
        self.omegaframe = omegaframe
        super(StaticPowerlawVThetaIC, self).__init__(index=2, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            n = -(2 * parameters["flaringindex"] - 1 - parameters["sigmaslope"])
            h = onet_disk2D.physics.pde.h(parameters, y)
            ic_values = ((1 - n * h**2) / r) ** (1 / 2.0) - self.omegaframe * r
            return ic_values

        return f


class StaticRingVThetaIC(IC):
    def __init__(self, omegaframe, **kwargs):
        self.omegaframe = omegaframe
        super(StaticRingVThetaIC, self).__init__(index=2, **kwargs)
        self.sigma_ic = RingSigmaIC(**kwargs)

    @functools.cached_property
    def func(self):
        # todo check the equation (with the Mathematica file)
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            sigma_ic = self.sigma_ic.func(parameters, y)
            cs = parameters["aspectratio"] * r ** (-0.5 + parameters["flaringindex"])
            # p = cs ** 2 * sigma_ic

            dsigma_dr = (
                (sigma_ic - parameters["sigma0"])
                * -(r - parameters["ringcenter"])
                / parameters["ringwidth"] ** 2
            )
            dcs_dr = (-0.5 + parameters["flaringindex"]) * cs / r
            dp_dr = cs**2 * dsigma_dr + 2 * cs * dcs_dr * sigma_ic

            ic_values = (r / sigma_ic * dp_dr + 1 / r) ** 0.5 - self.omegaframe * r
            return ic_values

        return f


class FungVThetaIC(IC):
    def __init__(self, omegaframe, **kwargs):
        self.omegaframe = omegaframe
        super(FungVThetaIC, self).__init__(index=2, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            h_over_r = parameters["aspectratio"] * r ** parameters["flaringindex"]
            factor = 1.0 + parameters["sigmaslope"] - 2.0 * parameters["flaringindex"]
            ic_values = (1 - factor * h_over_r**2) ** 0.5 * r ** (
                -0.5
            ) - self.omegaframe * r
            return ic_values

        return f


class StaticVRIC(IC):
    def __init__(self, **kwargs):
        super(StaticVRIC, self).__init__(index=1, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            return jnp.zeros_like(parameters["aspectratio"] * r)

        return f


class KeplerianRingVRIC(IC):
    def __init__(self, **kwargs):
        super(KeplerianRingVRIC, self).__init__(index=1, **kwargs)

    # todo check the equation (with the Mathematica file)
    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            inverse_exp_part = jnp.exp(
                -((r - parameters["ringcenter"]) ** 2)
                / (2.0 * parameters["ringwidth"] ** 2)
            )
            part = parameters["ringwidth"] ** 2 * (
                jnp.sqrt(2.0) * inverse_exp_part
                + 2.0 * jnp.sqrt(jnp.pi) * parameters["ringwidth"]
            )
            ic_values = (
                3.0
                * parameters["nu"]
                * (
                    (
                        2.0 * jnp.sqrt(2.0) * r**2
                        - 2.0 * jnp.sqrt(2.0) * r * parameters["ringcenter"]
                    )
                    * inverse_exp_part
                    - part
                )
                / (2.0 * r * part)
            )
            return ic_values

        return f


class FungVRIC(IC):
    def __init__(self, **kwargs):
        super(FungVRIC, self).__init__(index=1, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(parameters, y):
            """

            Args:
                parameters: A dict of problem-specific physics parameters.
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            h_over_r = parameters["aspectratio"] * r ** parameters["flaringindex"]
            ic_values = -1.5 * parameters["alpha"] * h_over_r**2 * r ** (-0.5)

            return ic_values

        return f


def get_sigma_ic(fargo_setups: dict):
    if fargo_setups["densityinitial"] == "RING2DDENS":
        sigma_ic = RingSigmaIC(**fargo_setups)
    elif fargo_setups["densityinitial"] == "POWERLAW2DDENS":
        sigma_ic = PowerlawSigmaIC(**fargo_setups)
    else:
        raise ValueError(f"sigma_type = {fargo_setups['densityinitial']} not found.")
    return sigma_ic


def get_v_r_ic(fargo_setups: dict):
    if fargo_setups["vyinitial"] == "KEPLERIANRINGVY":
        v_r_ic = KeplerianRingVRIC(**fargo_setups)
    elif fargo_setups["vyinitial"] == "STATICVY":
        v_r_ic = StaticVRIC()
    elif fargo_setups["vyinitial"] == "FUNG2DVY":
        v_r_ic = FungVRIC(**fargo_setups)
    else:
        raise ValueError(f"v_r_type = {fargo_setups['vyinitial']} not found.")
    return v_r_ic


def get_v_theta_ic(fargo_setups: dict):
    if fargo_setups["vxinitial"] == "KEPLERIAN2DVAZIM":
        v_theta_ic = KeplerianVThetaIC(**fargo_setups)
    elif fargo_setups["vxinitial"] == "STATICPOWERLAW2DVAZIM":
        v_theta_ic = StaticPowerlawVThetaIC(**fargo_setups)
    elif fargo_setups["vxinitial"] == "STATICRING2DVAZIM":
        v_theta_ic = StaticRingVThetaIC(**fargo_setups)
    elif fargo_setups["vxinitial"] == "FUNG2DVAZIM":
        v_theta_ic = FungVThetaIC(**fargo_setups)
    else:
        raise ValueError(f"v_theta_type = {fargo_setups['vxinitial']} not found.")
    return v_theta_ic


def get_transformed_s_fn(ic, s_fn):
    """

    Args:
        ic:
        s_fn: Callable(params, state, inputs)

    Returns:
        Callable(params, state, parameters, inputs)
    """

    @jax.jit
    def new_s_fn(params, state, parameters, inputs):
        outputs = s_fn(params, state, inputs)
        y = inputs["y_net"]
        s_initial = ic.func(parameters, y) if ic else jnp.zeros(outputs.shape)

        return s_initial + outputs

    return new_s_fn
