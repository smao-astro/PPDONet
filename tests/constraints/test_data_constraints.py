import pathlib

import jax.numpy as jnp
import jax.random
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

    @pytest.fixture(params=["log_sigma", "sigma"])
    def unknown(self, request):
        return request.param

    @pytest.fixture(params=[SINGLEP_DATAPATH, MULTIP_DATAPATH])
    def train_data(self, unknown, request):
        datapath = request.param
        datapath = pathlib.Path(datapath)
        return onet_disk2D.data.load_last_frame_data(datapath, unknown=unknown)

    @pytest.fixture
    def scaling_factors(self):
        return {"scaling_factors": jnp.array(0.1)}

    @pytest.fixture
    def model(self, n_node, y_net_layer_size, u_net_layer_size, unknown, train_data):
        parameter_name = onet_disk2D.data.extract_variable_parameters_name(
            train_data[unknown]
        )
        return onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            Nx=len(parameter_name),
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
    def data_constraints(self, s_fn, unknown, train_data):
        total_size = len(train_data[unknown]["run"])
        random_index_iterator = onet_disk2D.data.RandomIndexIterator(
            total_size=total_size, batch_size=1
        )
        return onet_disk2D.constraints.DataConstraints(
            s_fn,
            train_data=train_data,
            random_index_iterator=random_index_iterator,
        )

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
        assert True

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
