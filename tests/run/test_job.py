import pathlib
import shutil

import jax.numpy as jnp
import numpy as np
import pytest

import onet_disk2D.data_train
import onet_disk2D.run
import onet_disk2D.run
import onet_disk2D.train
import onet_disk2D.train

SINGLEP_DATAPATH = "../data/singlep"
MULTIP_DATAPATH = "../data/multip"


@pytest.fixture
def singlep_datapath(tmp_path):
    shutil.copytree(SINGLEP_DATAPATH, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def multip_datapath(tmp_path):
    shutil.copytree(MULTIP_DATAPATH, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture(params=[SINGLEP_DATAPATH, MULTIP_DATAPATH])
def datapath(request):
    return request.param


def test_load_fargo_setups(datapath):
    fargo_setups_file = pathlib.Path(datapath) / "fargo_setups.yml"
    fargo_setups, planet_config = onet_disk2D.run.job.load_fargo_setups(
        fargo_setups_file
    )
    assert np.isclose(fargo_setups["omegaframe"], 1.0)


def test_get_u_net_input_transform():
    col_idx_to_log = jnp.array([True, False, True])
    u_min = jnp.array([-4, 0.05, -4])
    u_max = jnp.array([-1, 0.1, -1])

    inputs = jnp.array(
        [
            [1e-4, 0.05, 1e-4],
            [1e-4, 0.1, 1e-4],
            [1e-1, 0.05, 1e-4],
            [1e-1, 0.1, 1e-4],
            [1e-4, 0.05, 1e-1],
            [1e-4, 0.1, 1e-1],
            [1e-1, 0.05, 1e-1],
            [1e-1, 0.1, 1e-1],
        ]
    )

    expected = jnp.array(
        [
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ]
    )
    u_net_input_transform = onet_disk2D.run.job.get_u_net_input_transform(
        col_idx_to_log, u_min, u_max
    )
    transformed_inputs = u_net_input_transform(inputs)

    assert jnp.allclose(transformed_inputs, expected)
