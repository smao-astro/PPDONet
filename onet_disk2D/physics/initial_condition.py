import functools
import typing

import jax
import jax.numpy as jnp


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
        def f(u, y):
            """

            Args:
                u: (Nu, Nx)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """

        return f


class RingSigmaIC(IC):
    def __init__(self, sigma0, ringcenter, ringwidth, **kwargs):
        super(RingSigmaIC, self).__init__(index=0, **kwargs)
        self.sigma0 = sigma0 if isinstance(sigma0, tuple) else float(sigma0)
        self.ringcenter = (
            ringcenter if isinstance(ringcenter, tuple) else float(ringcenter)
        )
        self.ringwidth = ringwidth if isinstance(ringwidth, tuple) else float(ringwidth)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            sigma0 = (
                u[..., self.sigma0[0], None]
                if isinstance(self.sigma0, tuple)
                else self.sigma0
            )
            ringcenter = (
                u[..., self.ringcenter[0], None]
                if isinstance(self.ringcenter, tuple)
                else self.ringcenter
            )
            ringwidth = (
                u[..., self.ringwidth[0], None]
                if isinstance(self.ringwidth, tuple)
                else self.ringwidth
            )
            ic_values = sigma0 * (
                1.0
                + jnp.exp(-((r - ringcenter) ** 2) / (2 * ringwidth**2))
                / (jnp.sqrt(2 * jnp.pi) * ringwidth)
            )
            return ic_values

        return f


class PowerlawSigmaIC(IC):
    def __init__(self, sigma0, sigmaslope, **kwargs):
        super(PowerlawSigmaIC, self).__init__(index=0, **kwargs)
        self.sigma0 = sigma0 if isinstance(sigma0, tuple) else float(sigma0)
        self.sigmaslope = (
            sigmaslope if isinstance(sigmaslope, tuple) else float(sigmaslope)
        )

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            sigma0 = (
                u[..., self.sigma0[0], None]
                if isinstance(self.sigma0, tuple)
                else self.sigma0
            )
            sigmaslope = (
                u[..., self.sigmaslope[0], None]
                if isinstance(self.sigmaslope, tuple)
                else self.sigmaslope
            )
            ic_values = sigma0 * r**-sigmaslope
            return ic_values

        return f


class KeplerianVThetaIC(IC):
    def __init__(self, omegaframe, **kwargs):
        self.omegaframe = omegaframe
        super(KeplerianVThetaIC, self).__init__(index=2, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            out = r**-0.5 - self.omegaframe * r
            if u.ndim > 1:
                return jnp.broadcast_to(out, (u.shape[0], r.shape[-1]))
            else:
                return out

        return f


class StaticPowerlawVThetaIC(IC):
    def __init__(self, omegaframe, sigmaslope, aspectratio, flaringindex, **kwargs):
        self.omegaframe = omegaframe
        self.sigmaslope = (
            sigmaslope if isinstance(sigmaslope, tuple) else float(sigmaslope)
        )
        self.aspectratio = (
            aspectratio if isinstance(aspectratio, tuple) else float(aspectratio)
        )
        self.flaringindex = (
            flaringindex if isinstance(flaringindex, tuple) else float(flaringindex)
        )
        super(StaticPowerlawVThetaIC, self).__init__(index=2, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            flaringindex = (
                u[..., self.flaringindex[0], None]
                if isinstance(self.flaringindex, tuple)
                else self.flaringindex
            )
            sigmaslope = (
                u[..., self.sigmaslope[0], None]
                if isinstance(self.sigmaslope, tuple)
                else self.sigmaslope
            )
            aspectratio = (
                u[..., self.aspectratio[0], None]
                if isinstance(self.aspectratio, tuple)
                else self.aspectratio
            )

            n = -(2 * flaringindex - 1 - sigmaslope)
            h = aspectratio * r**flaringindex
            ic_values = jnp.sqrt((1 - n * h**2) / r) - self.omegaframe * r
            return ic_values

        return f


class StaticRingVThetaIC(IC):
    def __init__(
        self,
        omegaframe,
        sigma0,
        ringcenter,
        ringwidth,
        aspectratio,
        flaringindex,
        **kwargs,
    ):
        self.omegaframe = omegaframe
        self.sigma0 = sigma0 if isinstance(sigma0, tuple) else float(sigma0)
        self.ringcenter = (
            ringcenter if isinstance(ringcenter, tuple) else float(ringcenter)
        )
        self.ringwidth = ringwidth if isinstance(ringwidth, tuple) else float(ringwidth)
        self.aspectratio = (
            aspectratio if isinstance(aspectratio, tuple) else float(aspectratio)
        )
        self.flaringindex = (
            flaringindex if isinstance(flaringindex, tuple) else float(flaringindex)
        )
        super(StaticRingVThetaIC, self).__init__(index=2, **kwargs)
        self.sigma_ic = RingSigmaIC(sigma0, ringcenter, ringwidth, **kwargs)

    @functools.cached_property
    def func(self):
        # todo check the equation (with the Mathematica file)
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            aspectratio = (
                u[..., self.aspectratio[0], None]
                if isinstance(self.aspectratio, tuple)
                else self.aspectratio
            )
            flaringindex = (
                u[..., self.flaringindex[0], None]
                if isinstance(self.flaringindex, tuple)
                else self.flaringindex
            )
            sigma0 = (
                u[..., self.sigma0[0], None]
                if isinstance(self.sigma0, tuple)
                else self.sigma0
            )
            ringcenter = (
                u[..., self.ringcenter[0], None]
                if isinstance(self.ringcenter, tuple)
                else self.ringcenter
            )
            ringwidth = (
                u[..., self.ringwidth[0], None]
                if isinstance(self.ringwidth, tuple)
                else self.ringwidth
            )

            sigma_ic = self.sigma_ic.func(u, y)
            cs = aspectratio * r ** (-0.5 + flaringindex)
            # p = cs ** 2 * sigma_ic

            dsigma_dr = (sigma_ic - sigma0) * -(r - ringcenter) / ringwidth**2
            dcs_dr = (-0.5 + flaringindex) * cs / r
            dp_dr = cs**2 * dsigma_dr + 2 * cs * dcs_dr * sigma_ic

            ic_values = (r / sigma_ic * dp_dr + 1 / r) ** 0.5 - self.omegaframe * r
            return ic_values

        return f


class StaticVRIC(IC):
    def __init__(self, **kwargs):
        super(StaticVRIC, self).__init__(index=1, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]
            return jnp.zeros_like(u[..., 0, None] * r)

        return f


class KeplerianRingVRIC(IC):
    def __init__(self, nu, ringcenter, ringwidth, **kwargs):
        super(KeplerianRingVRIC, self).__init__(index=1, **kwargs)
        self.nu = nu if isinstance(nu, tuple) else float(nu)
        self.ringcenter = (
            ringcenter if isinstance(ringcenter, tuple) else float(ringcenter)
        )
        self.ringwidth = ringwidth if isinstance(ringwidth, tuple) else float(ringwidth)

    # todo check the equation (with the Mathematica file)
    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            ringcenter = (
                u[..., self.ringcenter[0], None]
                if isinstance(self.ringcenter, tuple)
                else self.ringcenter
            )
            ringwidth = (
                u[..., self.ringwidth[0], None]
                if isinstance(self.ringwidth, tuple)
                else self.ringwidth
            )
            nu = u[..., self.nu[0], None] if isinstance(self.nu, tuple) else self.nu

            inverse_exp_part = jnp.exp(
                -((r - ringcenter) ** 2) / (2.0 * ringwidth**2)
            )
            part = ringwidth**2 * (
                jnp.sqrt(2.0) * inverse_exp_part + 2.0 * jnp.sqrt(jnp.pi) * ringwidth
            )
            ic_values = (
                3.0
                * nu
                * (
                    (
                        2.0 * jnp.sqrt(2.0) * r**2
                        - 2.0 * jnp.sqrt(2.0) * r * ringcenter
                    )
                    * inverse_exp_part
                    - part
                )
                / (2.0 * r * part)
            )
            return ic_values

        return f


class FungVRIC(IC):
    def __init__(self, alpha, aspectratio, flaringindex, sigmaslope, **kwargs):
        self.alpha = alpha if isinstance(alpha, tuple) else float(alpha)
        self.aspectratio = (
            aspectratio if isinstance(aspectratio, tuple) else float(aspectratio)
        )
        self.flaringindex = (
            flaringindex if isinstance(flaringindex, tuple) else float(flaringindex)
        )
        self.sigmaslope = (
            sigmaslope if isinstance(sigmaslope, tuple) else float(sigmaslope)
        )
        super(FungVRIC, self).__init__(index=1, **kwargs)

    @functools.cached_property
    def func(self):
        @jax.jit
        def f(u, y):
            """

            Args:
                u: (Nu, Np)
                y: spatial(-temporal) coordinates

            Returns:
                (Nu, Ny)
            """
            r = y[..., 0]

            alpha = (
                u[..., self.alpha[0], None]
                if isinstance(self.alpha, tuple)
                else self.alpha
            )
            aspectratio = (
                u[..., self.aspectratio[0], None]
                if isinstance(self.aspectratio, tuple)
                else self.aspectratio
            )
            flaringindex = (
                u[..., self.flaringindex[0], None]
                if isinstance(self.flaringindex, tuple)
                else self.flaringindex
            )
            sigmaslope = (
                u[..., self.sigmaslope[0], None]
                if isinstance(self.sigmaslope, tuple)
                else self.sigmaslope
            )

            h_over_r = aspectratio * r**flaringindex
            # only applies to cases where alpha and h/r are constants
            ic_values = -3 * (1 - sigmaslope) * alpha * h_over_r**2 * r ** (-0.5)

            return ic_values

        return f


def get_sigma_ic(densityinitial, parameters: dict):
    """

    Args:
        densityinitial:
        parameters: If the parameter is one of the inputs, the value should be a tuple specifying the index in `u_net`; otherwise, it should be a float.

    Returns:

    """
    if densityinitial == "RING2DDENS":
        sigma_ic = RingSigmaIC(**parameters)
    elif densityinitial == "POWERLAW2DDENS":
        sigma_ic = PowerlawSigmaIC(**parameters)
    else:
        raise ValueError(f"sigma_type = {densityinitial} not found.")
    return sigma_ic


def get_v_r_ic(vyinitial, parameters: dict):
    """

    Args:
        vyinitial:
        parameters: If the parameter is one of the inputs, the value should be a tuple specifying the index in `u_net`; otherwise, it should be a float.


    Returns:

    """
    if vyinitial == "KEPLERIANRINGVY":
        v_r_ic = KeplerianRingVRIC(**parameters)
    elif vyinitial == "STATICVY":
        v_r_ic = StaticVRIC()
    elif vyinitial == "FUNG2DVY":
        v_r_ic = FungVRIC(**parameters)
    else:
        raise ValueError(f"v_r_type = {vyinitial} not found.")
    return v_r_ic


def get_v_theta_ic(vxinitial, parameters: dict):
    """

    Args:
        vxinitial:
        parameters: If the parameter is one of the inputs, the value should be a tuple specifying the index in `u_net`; otherwise, it should be a float.

    Returns:

    """
    if vxinitial == "KEPLERIAN2DVAZIM":
        v_theta_ic = KeplerianVThetaIC(**parameters)
    elif vxinitial in ["STATICPOWERLAW2DVAZIM", "FUNG2DVAZIM"]:
        v_theta_ic = StaticPowerlawVThetaIC(**parameters)
    elif vxinitial == "STATICRING2DVAZIM":
        v_theta_ic = StaticRingVThetaIC(**parameters)
    # elif fargo_setups["vxinitial"] == "FUNG2DVAZIM":
    #     v_theta_ic = FungVThetaIC(**fargo_setups)
    else:
        raise ValueError(f"v_theta_type = {vxinitial} not found.")
    return v_theta_ic


# todo test
def get_transformed_s_fn(ic: IC, s_fn):
    """

    Args:
        ic: IC
        s_fn: Callable(params, state, inputs)

    Returns:
        Callable(params, state, inputs)
    """

    @jax.jit
    def new_s_fn(params, state, inputs):
        outputs = s_fn(params, state, inputs)
        s_initial = ic.func(inputs["u_net"], inputs["y_net"])
        return s_initial + outputs

    return new_s_fn
