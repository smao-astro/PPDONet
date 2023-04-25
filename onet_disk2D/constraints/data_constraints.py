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
import typing

import jax
import jax.numpy as jnp
import xarray as xr

from onet_disk2D.data import RandomIndexIterator, to_datadict
from .constraints import Constraints


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
            """data: {'inputs': {'y_net': y_net, 'u_net': y_net}, 's': s}"""
            s = self.s_fn(*args[:-1], data["inputs"])
            return s - data["s"]

        return f

    @functools.cached_property
    def loss_fn(self):
        @jax.jit
        def f(*args):
            return jnp.mean(self.res_fn(*args) ** 2)

        return f


class DataConstraints(Constraints):
    def __init__(
        self,
        s_pred_fn,
        train_data: typing.Mapping[str, xr.DataArray],
        random_index_iterator: RandomIndexIterator,
        **kwargs,
    ):
        """

        Args:
            s_pred_fn:
            train_data: a dict, key might include "log_sigma", "sigma", "v_r", or "v_theta"
            random_index_iterator:
            **kwargs:

        Notes:
            all element in `train_data` should have the same number of fargo runs, and share the same `random_index_iterator`
        """
        super(DataConstraints, self).__init__(s_pred_fn=s_pred_fn, **kwargs)
        self.s_pred_fn = s_pred_fn
        self.train_data = train_data
        self.random_index_iterator = random_index_iterator

    def samplers(self):
        pass

    @functools.cached_property
    def data_losses(self):
        """

        Returns:
            A dict, key is data_ + unknown

        Notes:
            currently only support one k: "log_sigma", "sigma", "v_r", or "v_theta". For multiple k, we need an s_pred_fn that output multiple values simultaneously, or a dict of s_pred_fn.
        """
        return {"data_" + k: DataLoss(self.s_pred_fn) for k in self.train_data}

    @functools.cached_property
    def loss_fn(self):
        """

        Returns:
            A dict, key is data_ + unknown
        """
        fn = super(DataConstraints, self).loss_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: data_loss.loss_fn for k, data_loss in self.data_losses.items()})
        return fn

    @functools.cached_property
    def res_fn(self):
        """

        Returns:
            A dict, key is data_ + unknown
        """
        fn = super(DataConstraints, self).res_fn
        # the last arguments: data {inputs: {u_net, y_net}, s}
        fn.update({k: data_loss.res_fn for k, data_loss in self.data_losses.items()})
        return fn

    def resample(self, key):
        super(DataConstraints, self).resample(key)
        indices = self.random_index_iterator.get_batch_indices()
        for k, v in self.train_data.items():
            v: xr.DataArray
            self.samples["data_" + k] = to_datadict(v.isel(run=indices))
