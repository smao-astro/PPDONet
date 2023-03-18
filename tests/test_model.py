import jax.numpy as jnp
import numpy as np
import pytest

import onet_disk2D.model
import onet_disk2D.physics.initial_condition as IC


class TestMLPONet:
    @pytest.fixture
    def activation(self):
        return onet_disk2D.model.get_activation("tanh")

    @pytest.fixture
    def initializer(self):
        return onet_disk2D.model.get_initializer("glorot_uniform")

    @pytest.fixture
    def Nx(self):
        return 3

    @pytest.fixture
    def Nu(self):
        return 7

    @pytest.fixture
    def Ny(self):
        return 11

    @pytest.fixture
    def Ndim(self):
        return 1

    @pytest.fixture
    def mlponet(self, Nx, Ndim, activation, initializer):
        net = onet_disk2D.model.MLPSingleONet(
            Nx,
            Ndim,
            [10, 20],
            activation,
            initializer,
        )
        net.build()

        return net

    @pytest.fixture
    def single_u_inputs(self, Nx, Ny, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_y_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        y = jnp.stack([y] * Nu)
        return {"u_net": u, "y_net": y}

    def test_inputs_shape(
        self, Nu, Ny, Nx, Ndim, single_u_inputs, multi_u_inputs, multi_u_y_inputs
    ):
        cri = [
            single_u_inputs["u_net"].shape == (Nx,),
            single_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_inputs["u_net"].shape == (Nu, Nx),
            multi_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_y_inputs["u_net"].shape == (Nu, Nx),
            multi_u_y_inputs["y_net"].shape == (Nu, Ny, Ndim),
        ]
        assert all(cri)

    def test_direct_apply(
        self,
        Nu,
        Ny,
        Nx,
        Ndim,
        single_u_inputs,
        multi_u_inputs,
        multi_u_y_inputs,
        mlponet,
    ):
        s_fn = mlponet.forward_apply
        single_u_s = s_fn(mlponet.params, single_u_inputs)
        multi_u_s = s_fn(mlponet.params, multi_u_inputs)
        multi_u_y_s = s_fn(mlponet.params, multi_u_y_inputs)
        cri = [
            single_u_s.shape == (Ny,),
            multi_u_s.shape == (Nu, Ny),
            multi_u_y_s.shape == (Nu, Ny),
        ]
        assert all(cri)

    def test_scaling_transform(
        self, mlponet: onet_disk2D.model.MLPSingleONet, multi_u_inputs
    ):
        s1 = mlponet.forward_apply(mlponet.params, multi_u_inputs)

        scaling_factors = {"scaling_factors": jnp.array([10.0])}
        s2_fn, s2_a_fn = onet_disk2D.model.outputs_scaling_transform(
            mlponet.forward_apply
        )

        s2 = s2_fn(mlponet.params, scaling_factors, multi_u_inputs)
        s2_, a = s2_a_fn(mlponet.params, scaling_factors, multi_u_inputs)

        cri = [
            jnp.all(jnp.isclose(s1 * 10, s2)),
            jnp.array_equal(s2, s2_),
        ]

        assert all(cri)


class TestDeepONet:
    @pytest.fixture
    def activation(self):
        return onet_disk2D.model.get_activation("tanh")

    @pytest.fixture
    def initializer(self):
        return onet_disk2D.model.get_initializer("glorot_uniform")

    @pytest.fixture
    def Nx(self):
        return 3

    @pytest.fixture
    def Nu(self):
        return 7

    @pytest.fixture
    def Ny(self):
        return 11

    @pytest.fixture
    def Ndim(self):
        return 1

    @pytest.fixture
    def Nnode(self):
        return 10

    @pytest.fixture
    def deeponet(self, activation, initializer, Nx, Ndim, Nnode):
        u_net = onet_disk2D.model.MLP(
            inputs_dim=Nx,
            outputs_dim=Nnode,
            layer_size=[10, 20],
            activation=activation,
            w_init=initializer,
        )
        u_net.build()
        y_net = onet_disk2D.model.MLP(
            inputs_dim=Ndim,
            outputs_dim=Nnode,
            layer_size=[30, 40],
            activation=activation,
            w_init=initializer,
        )
        y_net.build()

        net = onet_disk2D.model.DeepONet(u_net, y_net)

        return net

    @pytest.fixture
    def single_u_inputs(self, Nx, Ny, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_y_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        y = jnp.stack([y] * Nu)
        return {"u_net": u, "y_net": y}

    def test_inputs_shape(
        self, Nu, Ny, Nx, Ndim, single_u_inputs, multi_u_inputs, multi_u_y_inputs
    ):
        cri = [
            single_u_inputs["u_net"].shape == (Nx,),
            single_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_inputs["u_net"].shape == (Nu, Nx),
            multi_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_y_inputs["u_net"].shape == (Nu, Nx),
            multi_u_y_inputs["y_net"].shape == (Nu, Ny, Ndim),
        ]
        assert all(cri)

    def test_direct_apply(
        self,
        Nu,
        Ny,
        Nx,
        Ndim,
        single_u_inputs,
        multi_u_inputs,
        multi_u_y_inputs,
        deeponet,
    ):
        s_fn = deeponet.forward_apply
        single_u_s = s_fn(deeponet.params, single_u_inputs)
        multi_u_s = s_fn(deeponet.params, multi_u_inputs)
        multi_u_y_s = s_fn(deeponet.params, multi_u_y_inputs)
        cri = [
            single_u_s.shape == (Ny,),
            multi_u_s.shape == (Nu, Ny),
            multi_u_y_s.shape == (Nu, Ny),
        ]
        assert all(cri)

    def test_scaling_transform(
        self, deeponet: onet_disk2D.model.DeepONet, multi_u_inputs
    ):
        s1 = deeponet.forward_apply(deeponet.params, multi_u_inputs)

        scaling_factors = {"scaling_factors": jnp.array([10.0])}
        s2_fn, s2_a_fn = onet_disk2D.model.outputs_scaling_transform(
            deeponet.forward_apply
        )

        s2 = s2_fn(deeponet.params, scaling_factors, multi_u_inputs)
        s2_, a = s2_a_fn(deeponet.params, scaling_factors, multi_u_inputs)

        cri = [
            jnp.all(jnp.isclose(s1 * 10, s2)),
            jnp.array_equal(s2, s2_),
        ]

        assert all(cri)


class TestTriDeepONet:
    @pytest.fixture
    def Nx(self):
        return 3

    @pytest.fixture
    def Nu(self):
        return 7

    @pytest.fixture
    def Ny(self):
        return 11

    @pytest.fixture
    def Ndim(self):
        return 1

    @pytest.fixture
    def Nnode(self):
        return 10

    @pytest.fixture
    def deeponet(self, Nx, Ndim, Nnode):
        activation = onet_disk2D.model.get_activation("tanh")
        initializer = onet_disk2D.model.get_initializer("glorot_uniform")

        u_net = onet_disk2D.model.MLP(
            inputs_dim=Nx,
            outputs_dim=Nnode,
            layer_size=[10, 20],
            activation=activation,
            w_init=initializer,
        )
        u_net.build()
        y_net = onet_disk2D.model.MLP(
            inputs_dim=Ndim,
            outputs_dim=Nnode,
            layer_size=[30, 40],
            activation=activation,
            w_init=initializer,
        )
        y_net.build()
        z_net = onet_disk2D.model.MLP(
            inputs_dim=Nnode,
            outputs_dim=1,
            layer_size=[5],
            activation=activation,
            w_init=initializer,
        )
        z_net.build()

        net = onet_disk2D.model.TriDeepONet(u_net, y_net, z_net)

        return net

    @pytest.fixture
    def single_u_inputs(self, Nx, Ny, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        return {"u_net": u, "y_net": y}

    @pytest.fixture
    def multi_u_y_inputs(self, Ny, Nu, Nx, Ndim):
        u = jnp.arange(Nx, dtype=jnp.float32)
        u = jnp.stack([u] * Nu)
        y = jnp.stack([jnp.arange(Ny, dtype=jnp.float32)] * Ndim, axis=-1)
        y = jnp.stack([y] * Nu)
        return {"u_net": u, "y_net": y}

    def test_inputs_shape(
        self, Nu, Ny, Nx, Ndim, single_u_inputs, multi_u_inputs, multi_u_y_inputs
    ):
        cri = [
            single_u_inputs["u_net"].shape == (Nx,),
            single_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_inputs["u_net"].shape == (Nu, Nx),
            multi_u_inputs["y_net"].shape == (Ny, Ndim),
            multi_u_y_inputs["u_net"].shape == (Nu, Nx),
            multi_u_y_inputs["y_net"].shape == (Nu, Ny, Ndim),
        ]
        assert all(cri)

    def test_direct_apply(
        self,
        Nu,
        Ny,
        Nx,
        Ndim,
        single_u_inputs,
        multi_u_inputs,
        multi_u_y_inputs,
        deeponet,
    ):
        s_fn = deeponet.forward_apply
        single_u_s = s_fn(deeponet.params, single_u_inputs)
        multi_u_s = s_fn(deeponet.params, multi_u_inputs)
        multi_u_y_s = s_fn(deeponet.params, multi_u_y_inputs)
        cri = [
            single_u_s.shape == (Ny,),
            multi_u_s.shape == (Nu, Ny),
            multi_u_y_s.shape == (Nu, Ny),
        ]
        assert all(cri)

    def test_scaling_transform(
        self, deeponet: onet_disk2D.model.TriDeepONet, multi_u_inputs
    ):
        s1 = deeponet.forward_apply(deeponet.params, multi_u_inputs)

        scaling_factors = {"scaling_factors": jnp.array([10.0])}
        s2_fn, s2_a_fn = onet_disk2D.model.outputs_scaling_transform(
            deeponet.forward_apply
        )

        s2 = s2_fn(deeponet.params, scaling_factors, multi_u_inputs)
        s2_, a = s2_a_fn(deeponet.params, scaling_factors, multi_u_inputs)

        cri = [
            jnp.all(jnp.isclose(s1 * 10, s2)),
            jnp.array_equal(s2, s2_),
        ]

        assert all(cri)


def test_get_period_transform():
    assert False


def test_get_input_transform():
    assert False


class TestBuildModel:
    @pytest.fixture
    def u_net_layer_size(self):
        return [10, 20]

    @pytest.fixture
    def y_net_layer_size(self):
        return [15, 30]

    @pytest.fixture
    def n_node(self):
        return 4

    @pytest.fixture
    def u(self):
        """Parameter PLANETMASS"""
        return jnp.linspace(1e-4, 1e-6, 5)[:, None]

    @pytest.fixture
    def y(self):
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
        return y

    def test_y(self, y):
        assert y.shape == (15, 2)

    @pytest.fixture
    def inputs(self, u, y):
        return {"u_net": u, "y_net": y}

    def test_vanilla_model(self, u_net_layer_size, y_net_layer_size, n_node, inputs):
        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
        )
        s = model.forward_apply(model.params, inputs)
        assert s.shape == (inputs["u_net"].shape[0], inputs["y_net"].shape[0])

    def test_periodic_transform_model(
        self, u_net_layer_size, y_net_layer_size, n_node, inputs
    ):
        periodic_transform = onet_disk2D.model.get_period_transform(0.4, 2.5)

        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            y_net_input_transform=periodic_transform,
        )
        s = model.forward_apply(model.params, inputs)
        assert s.shape == (inputs["u_net"].shape[0], inputs["y_net"].shape[0])

    def test_input_normalization_model(
        self, u_net_layer_size, y_net_layer_size, n_node, inputs
    ):
        u_min = jnp.min(inputs["u_net"], axis=0)
        u_max = jnp.max(inputs["u_net"], axis=0)
        u_transform = onet_disk2D.model.get_input_normalization(
            u_min=u_min, u_max=u_max
        )

        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            u_net_input_transform=u_transform,
        )
        s = model.forward_apply(model.params, inputs)
        assert s.shape == (inputs["u_net"].shape[0], inputs["y_net"].shape[0])

    # test outputs_scaling_transform
    def test_outputs_scaling_transform_model(
        self, u_net_layer_size, y_net_layer_size, n_node, inputs
    ):
        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
        )
        s = model.forward_apply(model.params, inputs)

        scaling_factors = {"scaling_factors": jnp.array(10.0)}
        s2_fn = onet_disk2D.model.outputs_scaling_transform(model.forward_apply)[0]
        s2 = s2_fn(model.params, scaling_factors, inputs)

        assert jnp.all(jnp.isclose(s * scaling_factors["scaling_factors"], s2))

    @pytest.fixture
    def ic(self):
        fargo_setups = {
            "densityinitial": "POWERLAW2DDENS",
            "vyinitial": "STATICVY",
            "vxinitial": "STATICPOWERLAW2DVAZIM",
            "omegaframe": 1.0,
            "sigma0": 1.0,
            "sigmaslope": 0.5,
        }
        return IC.get_sigma_ic("POWERLAW2DDENS", fargo_setups)

    def test_ic_shifted_model(
        self, u_net_layer_size, y_net_layer_size, n_node, ic, inputs
    ):
        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
        )
        scaling_factors = {"scaling_factors": jnp.array(10.0)}
        s_fn = onet_disk2D.model.outputs_scaling_transform(model.forward_apply)[0]
        s = s_fn(model.params, scaling_factors, inputs)
        sigma0 = 1.0
        sigmaslope = 0.5
        ic_value = sigma0 * inputs["y_net"][..., 0] ** -sigmaslope

        s2_fn = IC.get_transformed_s_fn(ic, s_fn)
        s2 = s2_fn(model.params, scaling_factors, inputs)

        cri = [s.shape == s2.shape, jnp.all(jnp.isclose(s + ic_value, s2))]
        assert all(cri)

    def test_ic_shifted_model_mul_parameters(
        self,
        u_net_layer_size,
        y_net_layer_size,
        n_node,
        y,
    ):
        alpha = np.logspace(-4, -3, 10)[:, None]
        aspectratio = np.linspace(0.5, 1.0, 10)[:, None]
        flaringindex = 0.0
        sigmaslope = 0.5
        u = np.concatenate([alpha, aspectratio], axis=-1)
        inputs = {"u_net": u, "y_net": y}

        fargo_setups = {
            "alpha": (0,),
            "aspectratio": (1,),
            "flaringindex": "0.0",
            "sigmaslope": "0.5",
        }
        ic = IC.FungVRIC(**fargo_setups)

        model = onet_disk2D.model.build_model(
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            Nx=2,
            Nnode=n_node,
        )
        scaling_factors = {"scaling_factors": jnp.array(10.0)}
        s_fn = onet_disk2D.model.outputs_scaling_transform(model.forward_apply)[0]
        s = s_fn(model.params, scaling_factors, inputs)
        sigma0 = 1.0
        sigmaslope = 0.5

        r = inputs["y_net"][..., 0]
        h_over_r = aspectratio * r**flaringindex
        ic_value = -3 * (1 - sigmaslope) * alpha * h_over_r**2 * r ** (-0.5)

        s2_fn = IC.get_transformed_s_fn(ic, s_fn)
        s2 = s2_fn(model.params, scaling_factors, inputs)

        cri = [s.shape == s2.shape, jnp.all(jnp.isclose(s + ic_value, s2))]
        assert all(cri)

    def test_build_model_from_args(
        self, u_net_layer_size, y_net_layer_size, n_node, inputs
    ):
        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            unknown_parameter=1.0,
        )
        s = model.forward_apply(model.params, inputs)
        assert s.shape == (
            inputs["u_net"].shape[0],
            inputs["y_net"].shape[0],
        )
