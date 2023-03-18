import jax
import jax.numpy as jnp
import haiku as hk


class Stan(hk.Module):
    def __init__(self, beta_init=1.0):
        super(Stan, self).__init__(name="stan")
        self.beta_init = beta_init

    def __call__(self, inputs: jnp.ndarray):
        beta = hk.get_parameter(
            "stan_beta",
            [inputs.shape[-1]],
            init=hk.initializers.Constant(self.beta_init),
        )
        return (1.0 + beta * inputs) * jax.nn.tanh(inputs)


def stan(x):
    return Stan(beta_init=1.0)(x)


def get_activation(name: str):
    if name == "tanh":
        return jax.nn.tanh
    if name == "sin":
        return jnp.sin
    if name == "stan":
        return stan
    if name == "swish":
        return jax.nn.swish
    else:
        raise NotImplementedError
