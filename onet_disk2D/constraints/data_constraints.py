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
import jaxphyinf.constraints


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


class DataConstraints(jaxphyinf.constraints.Constraints):
    def __init__(
        self,
        s_pred_fn,
        unknown,
        dataloader,
        **kwargs,
    ):
        """

        Args:
            s_pred_fn:
            unknown: One of {'log_sigma', 'sigma', 'v_r', 'v_theta'}
            dataloader:
            ic: dict, One of {ic_sigma, ic_v_r, ic_v_theta}
            **kwargs:
        """
        super(DataConstraints, self).__init__(s_pred_fn=s_pred_fn, **kwargs)
        self.s_pred_fn = s_pred_fn
        self.unknown = unknown
        self.dataloader = dataloader
        self.data = iter(self.dataloader)

    def samplers(self):
        pass

    @functools.cached_property
    def dataloss(self):
        """

        Returns:
            A dict, key is data_ + unknown
        """
        return {"data_" + self.unknown: DataLoss(self.s_pred_fn)}

    @functools.cached_property
    def loss_fn(self):
        """

        Returns:
            A dict, key is data_ + unknown
        """
        fn = super(DataConstraints, self).loss_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: l.loss_fn for k, l in self.dataloss.items()})
        return fn

    @functools.cached_property
    def res_fn(self):
        """

        Returns:
            A dict, key is data_ + unknown
        """
        fn = super(DataConstraints, self).res_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: l.res_fn for k, l in self.dataloss.items()})
        return fn

    def resample(self, key):
        super(DataConstraints, self).resample(key)
        samples = next(self.data)
        for k, v in samples.items():
            self.samples["data_" + k] = v
