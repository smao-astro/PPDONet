import jax.numpy as jnp
import jaxphyinf.model
import numpy as np
import pytest

import onet_disk2D.model
import onet_disk2D.physics.initial_condition as IC


class TestModel:
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

    def test_output_transform_model(
        self, u_net_layer_size, y_net_layer_size, n_node, inputs
    ):
        def out_fn(outputs, inputs):
            return outputs * 10.0

        io_transform = jaxphyinf.model.input_output_transform(
            None, output_transform=out_fn
        )

        model = onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
        )

        s_fn = model.forward_apply
        s = s_fn(model.params, inputs)

        s2_fn = io_transform(model.forward_apply)
        s2 = s2_fn(model.params, inputs)

        assert jnp.all(jnp.isclose(s * 10.0, s2))

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
        s_fn = jaxphyinf.model.outputs_scaling_transform(model.forward_apply)[0]
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
