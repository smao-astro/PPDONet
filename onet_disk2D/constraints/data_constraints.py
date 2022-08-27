"""
constraints may include losses below:
    data: data loss
    pde: pde loss
    bc: boundary condition loss
Datasets for these losses vary among:
    sampling strategies
    data format
"""

import functools

import jax
import jax.numpy as jnp

from .constraints import ParametricConstraints


class DataLoss:
    def __init__(self, s_fn, **kwargs):
        """

        Args:
            s_fn:
        """
        self.s_fn = s_fn

    @functools.cached_property
    def res_fn(self):
        @jax.jit
        def f(*args):
            """

            Args:
                *args:

            Returns:
                shape (Nu, Ny) or (Ny,) or ()
            """
            data = args[-1]
            s = self.s_fn(*args[:-1], data["inputs"])
            return s - data["s"]

        return f

    @functools.cached_property
    def loss_fn(self):
        @jax.jit
        def f(*args):
            return jnp.mean(self.res_fn(*args) ** 2)

        return f


class WeightedDataLoss(DataLoss):
    def __init__(self, s_fn, ic_fn, data_loss_weighting: str, **kwargs):
        super(WeightedDataLoss, self).__init__(
            s_fn=s_fn,
            **kwargs,
        )
        self.ic_fn = ic_fn
        self.data_loss_weighting = data_loss_weighting
        self.low = 1.0
        self.high = 10.0

    @functools.cached_property
    def diff2_fn(self):
        """Weighting residuals by (s-s_ic)**2

        The weights are positively related to the divergence of ground truths from backgrounds.
        """

        @jax.jit
        def f(parameters, data):
            s_ic = self.ic_fn(parameters, data["inputs"]["y_net"])
            diff = (data["s"] - s_ic) ** 2
            width = jnp.max(diff, axis=-1, keepdims=True)
            diff = 50.0 * diff / width
            return (self.high - self.low) * (1 - jnp.exp(-diff)) + self.low

        return f

    @functools.cached_property
    def mag_fn(self):
        @jax.jit
        def f(parameters, data):
            s_ic = self.ic_fn(parameters, data["inputs"]["y_net"])
            # The squared values might be too large. -> Take square-root.
            mag = jnp.mean((data["s"] - s_ic) ** 2, axis=-1, keepdims=True)
            w = 1.0 / mag
            # w = w / jnp.sum(w)
            return w

        return f

    @functools.cached_property
    def w_fn(self):
        if self.data_loss_weighting == "diff2":
            return self.diff2_fn
        elif self.data_loss_weighting == "mag":
            return self.mag_fn
        else:
            raise NotImplementedError

    @functools.cached_property
    def loss_fn(self):
        @jax.jit
        def f(*args):
            parameters = args[-2]
            data = args[-1]
            w = self.w_fn(parameters, data)
            res2 = w * self.res_fn(*args) ** 2
            return jnp.mean(res2)

        return f


class DataConstraints(ParametricConstraints):
    def __init__(
        self,
        s_pred_fn,
        unknown,
        dataloader,
        ic=None,
        data_loss_weighting="",
        **kwargs,
    ):
        """

        Args:
            s_pred_fn:
            unknown: One of {'log_sigma', 'sigma', 'v_r', 'v_theta'}
            dataloader:
            ic: dict, One of {ic_sigma, ic_v_r, ic_v_theta}
            data_loss_weighting: '', 'diff2', 'mag',
            **kwargs:
        """
        super(DataConstraints, self).__init__(s_pred_fn=s_pred_fn, **kwargs)
        self.s_pred_fn = s_pred_fn
        self.unknown = unknown
        self.dataloader = dataloader
        self.ic = ic
        self.data_loss_weighting = data_loss_weighting
        self.data = iter(self.dataloader)

    def samplers(self):
        pass

    @functools.cached_property
    def dataloss(self):
        if not self.data_loss_weighting:
            loss = {"data_" + self.unknown: DataLoss(self.s_pred_fn)}
        elif self.data_loss_weighting in ["diff2", "mag"]:
            if self.unknown == "log_sigma":
                raise NotImplementedError
            loss = {
                "data_"
                + self.unknown: WeightedDataLoss(
                    s_fn=self.s_pred_fn,
                    ic_fn=self.ic.func,
                    data_loss_weighting=self.data_loss_weighting,
                )
            }
        else:
            raise NotImplementedError

        return loss

    @functools.cached_property
    def loss_fn(self):
        """

        Returns:
            A dict, key is one of {'log_sigma', 'sigma', 'v_r', 'v_theta'}
        """
        fn = super(DataConstraints, self).loss_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: l.loss_fn for k, l in self.dataloss.items()})
        return fn

    @functools.cached_property
    def res_fn(self):
        """

        Returns:
            A dict, key is one of {'log_sigma', 'sigma', 'v_r', 'v_theta'}
        """
        fn = super(DataConstraints, self).res_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: l.res_fn for k, l in self.dataloss.items()})
        return fn

    def resample(self, key):
        super(DataConstraints, self).resample(key)
        parameters, samples = next(self.data)
        for k, v in samples.items():
            self.parameters["data_" + k] = parameters
            self.samples["data_" + k] = v
