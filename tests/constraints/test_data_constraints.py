import pathlib

import jax.numpy as jnp
import jax.random
import jaxphyinf.model
import pytest

import onet_disk2D.constraints
import onet_disk2D.data
import onet_disk2D.model
import onet_disk2D.physics.initial_condition

SINGLEP_DATAPATH = "../data/singlep"
MULTIP_DATAPATH = "../data/multip"


class TestDataConstraints:
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

    @pytest.fixture(params=["log_sigma", "sigma"])
    def unknown(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, u, y):
        return {"u_net": u, "y_net": y}

    @pytest.fixture(params=[SINGLEP_DATAPATH, MULTIP_DATAPATH])
    def dataloader(self, unknown, request):
        datapath = request.param
        datapath = pathlib.Path(datapath)
        data = onet_disk2D.data.load_last_frame_data(datapath, unknown=unknown)
        batch_size = 9
        return onet_disk2D.data.DataIterLoader(data, batch_size)

    @pytest.fixture
    def scaling_factors(self):
        return {"scaling_factors": jnp.array(0.1)}

    @pytest.fixture
    def model(self, n_node, y_net_layer_size, u_net_layer_size, dataloader):
        return onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            Nx=len(dataloader.parameter_names),
        )

    @pytest.fixture
    def ic(self, unknown):
        if unknown == "sigma":
            return onet_disk2D.physics.initial_condition.PowerlawSigmaIC("1.0", "0.0")
        else:
            return None

    @pytest.fixture
    def s_fn(self, model, ic):
        f = onet_disk2D.model.outputs_scaling_transform(model.forward_apply)[0]
        if ic is not None:
            f = onet_disk2D.physics.initial_condition.get_transformed_s_fn(ic, f)
        return f

    @pytest.fixture
    def data_constraints(self, s_fn, unknown, dataloader):
        return onet_disk2D.constraints.DataConstraints(
            s_fn, unknown=unknown, dataloader=dataloader
        )

    def test_inputs(self, inputs):
        u = inputs["u_net"]
        y = inputs["y_net"]
        cri = [u.shape == (5, 1), y.shape == (15, 2)]
        assert all(cri)

    def test_loss_fn(self, model, scaling_factors, data_constraints):
        data_constraints.resample(123)
        loss = {
            k: loss_fn(
                model.params,
                scaling_factors,
                data_constraints.samples[k],
            )
            for k, loss_fn in data_constraints.loss_fn.items()
        }
        assert set(loss) == {"data_" + data_constraints.unknown}

    def test_res_fn(self, model, scaling_factors, data_constraints):
        key = jax.random.PRNGKey(999)

        data_constraints.resample(key)
        res = {
            k: fn(
                model.params,
                scaling_factors,
                data_constraints.samples[k],
            )
            for k, fn in data_constraints.res_fn.items()
        }
        cri = [
            v.shape == data_constraints.samples[k]["s"].shape for k, v in res.items()
        ]

        assert all(cri)
