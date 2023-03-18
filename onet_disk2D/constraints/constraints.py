import abc
import functools
from typing import TypedDict

import jax
import jax.numpy as jnp


class DataDict(TypedDict):
    inputs: jnp.ndarray
    y: jnp.ndarray


class Constraints(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.samples = {}

    @property
    @abc.abstractmethod
    def samplers(self):
        return {}

    @property
    @abc.abstractmethod
    def loss_fn(self):
        return {}

    @property
    @abc.abstractmethod
    def res_fn(self):
        return {}

    @functools.cached_property
    def v_g_fn(self):
        return {
            key: jax.jit(jax.value_and_grad(fn)) for key, fn in self.loss_fn.items()
        }

    @abc.abstractmethod
    def resample(self, key):
        pass

    def get_v_g(self, *args):
        """Compute loss values and gradients.

        Returns:
            (dict, dict)
        """
        v_g = {k: v_g_fn(*args, self.samples[k]) for k, v_g_fn in self.v_g_fn.items()}
        return {k: v[0] for k, v in v_g.items()}, {k: v[1] for k, v in v_g.items()}
