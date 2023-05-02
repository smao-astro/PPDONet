import abc
import functools

import haiku as hk
import jax
import jax.numpy as jnp

from .activation import get_activation
from .initialization import get_initializer


class NN(abc.ABC):
    def __init__(self):
        self.transformed = None
        self.params = None
        self.forward_apply = None
        self.input_transform = None
        self.output_transform = None

    @abc.abstractmethod
    def build(self):
        pass


class MLP(NN):
    def __init__(
        self,
        inputs_dim: int,
        outputs_dim: int,
        layer_size: list,
        activation,
        w_init,
        b_init=hk.initializers.Constant(0.0),
        key=jax.random.PRNGKey(123),
    ):
        """

        Args:
            layer_size: Number of neurons of hidden layers, do not include the dimension of outputs.
        """
        super(MLP, self).__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.layer_size = layer_size
        self.activation = activation
        self.w_init = w_init
        self.b_init = b_init
        self.key = key

    def build(self):
        @hk.without_apply_rng
        @hk.transform
        def forward_fn(inputs):
            if self.input_transform:
                outputs = self.input_transform(inputs)
            else:
                outputs = inputs

            model = hk.nets.MLP(
                output_sizes=self.layer_size + [self.outputs_dim],
                w_init=self.w_init,
                b_init=self.b_init,
                activation=self.activation,
            )
            outputs = model(outputs)
            if self.output_transform:
                outputs = self.output_transform(outputs, inputs)
            return outputs

        self.transformed = forward_fn
        self.params = forward_fn.init(self.key, jnp.zeros(shape=(1, self.inputs_dim)))
        self.forward_apply = jax.jit(forward_fn.apply)


class MLPSingleONet(NN):
    def __init__(
        self,
        Nx: int,
        Ndim: int,
        layer_size: list,
        activation,
        w_init,
        b_init=hk.initializers.Constant(0.0),
        key=jax.random.PRNGKey(123),
    ):
        """

        Args:
            Nx: last dimension of the u_net.
                If the deeponet maps one parameter to function, Nx=1 (default).
                If the deeponet maps a vector to function, Nx=len(u).
            Ndim: Dimension of coordinates. If y -> ('r', 'theta'), Ndim=2
            layer_size:
            activation:
            w_init:
            b_init:
            key: random seed for initializing network parameters
        """
        super(MLPSingleONet, self).__init__()
        self.Nx = Nx
        self.Ndim = Ndim
        self.inputs_dim = Nx + Ndim
        self.layer_size = layer_size
        self.activation = activation
        self.w_init = w_init
        self.b_init = b_init
        self.key = key

        self.u_net_input_transform = None
        self.y_net_input_transform = None

    def build(self):
        @hk.without_apply_rng
        @hk.transform
        def forward_fn(inputs: dict):
            """

            Args:
                inputs: {'u_net': [], 'y_net': []}

            Returns:

            """
            outputs = inputs.copy()
            if self.u_net_input_transform:
                outputs["u_net"] = self.u_net_input_transform(outputs["u_net"])
                outputs["y_net"] = self.y_net_input_transform(outputs["y_net"])

            nx = outputs["u_net"].shape[-1]
            ndim = outputs["y_net"].shape[-1]

            u_ndim = outputs["u_net"].ndim
            y_ndim = outputs["y_net"].ndim

            shape = []
            if u_ndim > 2:
                raise NotImplementedError
            elif u_ndim == 2:
                nu = outputs["u_net"].shape[0]
                shape.append(nu)

            if y_ndim > 1:
                ny = outputs["y_net"].shape[-2]
                shape.append(ny)

            if u_ndim > 1 and y_ndim > 1:
                outputs["u_net"] = jnp.expand_dims(outputs["u_net"], axis=-2)
            if len(shape) > 0:
                outputs["u_net"] = jnp.broadcast_to(outputs["u_net"], shape + [nx])
                outputs["y_net"] = jnp.broadcast_to(outputs["y_net"], shape + [ndim])
            outputs = jnp.concatenate([outputs["u_net"], outputs["y_net"]], axis=-1)

            model = hk.nets.MLP(
                output_sizes=self.layer_size + [1],
                w_init=self.w_init,
                b_init=self.b_init,
                activation=self.activation,
            )
            outputs = model(outputs)[..., 0]

            return outputs

        self.transformed = forward_fn
        self.params = forward_fn.init(
            self.key,
            {
                "u_net": jnp.zeros(shape=(1, self.Nx)),
                "y_net": jnp.zeros(shape=(1, self.Ndim)),
            },
        )
        self.forward_apply = jax.jit(forward_fn.apply)


class DeepONet:
    def __init__(self, u_net: NN, y_net: NN):
        self.u_net = u_net
        self.y_net = y_net

        self.inputs_dim = {}
        self.collect_attribute("inputs_dim")
        self.outputs_dim = {}
        self.collect_attribute("outputs_dim")
        if self.outputs_dim["u_net"] != self.outputs_dim["y_net"]:
            raise ValueError("outputs_dim do not match.")
        self.layer_size = {}
        self.collect_attribute("layer_size")

        self.params = {}
        self.collect_attribute("params")
        self.state = {}
        self.collect_attribute("state")

    def collect_attribute(self, name):
        for net in ["u_net", "y_net"]:
            if hasattr((n := getattr(self, net)), name):
                getattr(self, name).update({net: getattr(n, name)})

    @functools.cached_property
    def forward_apply(self):
        @jax.jit
        def f(*args):
            """

            Args:
                *args:
                    Dicts, with keys 'u_net', 'y_net', etc
                    ({'u_net': [], 'y_net': []}, {'u_net': [], 'y_net': []}, {'u_net': [], 'y_net': []})
            Returns:
                outputs. The last dimension is the index of unkowns (equal to one)
            """
            args_u_net = [v["u_net"] for v in args if "u_net" in v]
            args_y_net = [v["y_net"] for v in args if "y_net" in v]

            outputs_u_net = self.u_net.forward_apply(*args_u_net)
            outputs_y_net = self.y_net.forward_apply(*args_y_net)

            if outputs_y_net.ndim > 1:
                outputs_u_net_shape = outputs_u_net.shape[:-1] + (
                    1,
                    self.outputs_dim["u_net"],
                )
                outputs_u_net = outputs_u_net.reshape(outputs_u_net_shape)

            outputs = jnp.sum(outputs_u_net * outputs_y_net, axis=-1)

            return outputs

        return f


class TriDeepONet:
    def __init__(self, u_net: NN, y_net: NN, z_net: MLP):
        self.u_net = u_net
        self.y_net = y_net
        self.z_net = z_net

        if z_net.outputs_dim != 1:
            raise NotImplementedError

        self.inputs_dim = {}
        self.collect_attribute("inputs_dim")
        self.outputs_dim = {}
        self.collect_attribute("outputs_dim")
        if self.outputs_dim["u_net"] != self.outputs_dim["y_net"]:
            raise ValueError("outputs_dim do not match.")
        if self.inputs_dim["z_net"] != self.outputs_dim["u_net"]:
            raise ValueError("inputs_dim and outputs_dim do not match.")
        self.layer_size = {}
        self.collect_attribute("layer_size")

        self.params = {}
        self.collect_attribute("params")
        self.state = {}
        self.collect_attribute("state")

    def collect_attribute(self, name):
        for net in ["u_net", "y_net", "z_net"]:
            if hasattr((n := getattr(self, net)), name):
                getattr(self, name).update({net: getattr(n, name)})

    @functools.cached_property
    def forward_apply(self):
        @jax.jit
        def f(*args):
            """

            Args:
                *args:
                    Dicts, with keys 'u_net', 'y_net', etc
                    ({'u_net': [], 'y_net': []}, {'u_net': [], 'y_net': []}, {'u_net': [], 'y_net': []})
            Returns:
                outputs. The last dimension is the index of unkowns (equal to one)
            """
            args_u_net = [v["u_net"] for v in args if "u_net" in v]
            args_y_net = [v["y_net"] for v in args if "y_net" in v]
            args_z_net = [v["z_net"] for v in args if "z_net" in v]

            outputs_u_net = self.u_net.forward_apply(*args_u_net)
            outputs_y_net = self.y_net.forward_apply(*args_y_net)

            if outputs_y_net.ndim > 1:
                outputs_u_net_shape = outputs_u_net.shape[:-1] + (
                    1,
                    self.outputs_dim["u_net"],
                )
                outputs_u_net = outputs_u_net.reshape(outputs_u_net_shape)

            outputs = outputs_u_net * outputs_y_net
            outputs = self.z_net.forward_apply(*args_z_net, outputs)[..., 0]

            return outputs

        return f


def scale_to_one(u, u_min, u_max):
    u_middle = (u_min + u_max) / 2.0
    return (u - u_middle) / (u_max - u_min) * 2.0


# period boundary condition
def get_period_transform(r_min, r_max):
    def transform(inputs):
        r, theta = jnp.split(inputs, 2, -1)
        """(NUM_BATCH, 1)"""
        return jnp.concatenate(
            (
                scale_to_one(r, r_min, r_max),
                jnp.sin(theta),
                jnp.cos(theta),
            ),
            axis=-1,
        )

    return transform


def outputs_scaling_transform(f):
    @jax.jit
    def outputs_fn(*args):
        """

        Args:
            *args: (params, {'scaling_factors'}, inputs),
                or (params, state, scaling_factors, inputs)

        Returns:

        """
        scaling_factors = args[-2]["scaling_factors"]
        inputs = args[-1]

        outputs = f(*args[:-2], inputs)
        outputs = scaling_factors * outputs

        return outputs

    @jax.jit
    def outputs_and_a_fn(*args):
        """

        Args:
            *args: (params, scaling_factors, inputs),
                or (params, state, scaling_factors, inputs)

        Returns:

        """
        scaling_factors = args[-2]["scaling_factors"]
        inputs = args[-1]

        outputs = f(*args[:-2], inputs)
        a = jnp.mean(jnp.abs(outputs))

        outputs = scaling_factors * outputs

        return outputs, a

    return outputs_fn, outputs_and_a_fn


def get_input_normalization(u_min, u_max):
    @jax.jit
    def transform(inputs):
        mid = (u_min + u_max) / 2.0
        return 2.0 * (inputs - mid) / (u_max - u_min)

    return transform


def build_mlponet(
    layer_size,
    Nx=1,
    Ndim=2,
    activation="tanh",
    initializer="glorot_uniform",
    u_net_input_transform=None,
    y_net_input_transform=None,
    **kwargs,
):
    """Build an MLP model, but as an Operator Network"""
    activation = get_activation(activation)
    initializer = get_initializer(initializer)

    model = MLPSingleONet(
        Nx=Nx,
        Ndim=Ndim,
        layer_size=layer_size,
        activation=activation,
        w_init=initializer,
    )
    model.u_net_input_transform = u_net_input_transform
    model.y_net_input_transform = y_net_input_transform
    model.build()

    return model


def build_model(
    Nnode,
    u_net_layer_size,
    y_net_layer_size,
    Nx=1,
    Ndim=2,
    activation="tanh",
    initializer="glorot_uniform",
    u_net_input_transform=None,
    u_net_output_transform=None,
    y_net_input_transform=None,
    y_net_output_transform=None,
    z_net_layer_size=None,
    **kwargs,
):
    """Build DeepONet or TriDeepONet model

    Args:
        Nnode: Number of neurons of the last layer of u_net (and y_net).
        u_net_layer_size:
        y_net_layer_size:
        Nx: last dimension of the u_net.
            If the deeponet maps one parameter to function, Nx=1 (default).
            If the deeponet maps a vector to function, Nx=len(u).
        Ndim: Dimension of coordinates. If y -> ('r', 'theta'), Ndim=2
        activation:
        initializer:
        u_net_input_transform:
        u_net_output_transform:
        y_net_input_transform:
        y_net_output_transform:
        z_net_layer_size: If not empty list or None, use TriDeepONet

    Returns:

    """
    activation = get_activation(activation)
    initializer = get_initializer(initializer)

    u_net = MLP(
        inputs_dim=Nx,
        outputs_dim=Nnode,
        layer_size=u_net_layer_size,
        activation=activation,
        w_init=initializer,
    )
    # normalization
    u_net.input_transform = u_net_input_transform
    u_net.output_transform = u_net_output_transform
    u_net.build()

    y_net = MLP(
        inputs_dim=Ndim,
        outputs_dim=Nnode,
        layer_size=y_net_layer_size,
        activation=activation,
        w_init=initializer,
    )
    # periodic boundary
    # normalization
    y_net.input_transform = y_net_input_transform
    y_net.output_transform = y_net_output_transform

    y_net.build()

    if z_net_layer_size:
        z_net = MLP(
            inputs_dim=Nnode,
            outputs_dim=1,
            layer_size=z_net_layer_size,
            activation=activation,
            w_init=initializer,
        )
        z_net.build()
        model = TriDeepONet(u_net, y_net, z_net)
    else:
        # [2022.05.30] output transform can be done outside this function
        model = DeepONet(u_net, y_net)

    return model
