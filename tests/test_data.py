import pathlib

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import xarray as xr

import onet_disk2D.data
import onet_disk2D.run.job

SINGLEP_DATAPATH = "data/singlep"
MULTIP_DATAPATH = "data/multip"


@pytest.mark.parametrize(
    argnames="datapath", argvalues=[SINGLEP_DATAPATH, MULTIP_DATAPATH]
)
def test_load_last_frame_data(datapath):
    datapath = pathlib.Path(datapath)
    data = onet_disk2D.data.load_last_frame_data(datapath, unknown="log_sigma")
    sigma = xr.load_dataarray(datapath / "batch_truth_sigma.nc")
    log_sigma_ = np.log10(sigma)

    assert np.array_equal(log_sigma_.values, data["log_sigma"].values)


def test_extract_variable_parameters_name():
    datapath = pathlib.Path(MULTIP_DATAPATH)
    sigma = xr.load_dataarray(datapath / "batch_truth_sigma.nc")
    p_names = onet_disk2D.data.extract_variable_parameters_name(sigma)
    assert p_names == ["ASPECTRATIO", "PLANETMASS"]


@pytest.mark.parametrize(
    argnames="datapath", argvalues=[SINGLEP_DATAPATH, MULTIP_DATAPATH]
)
def test_to_datadict(datapath):
    data = xr.load_dataarray((f"{datapath}/batch_truth_sigma.nc"))
    parameters = list(set(data.coords) - set(data.dims))
    parameters.sort()
    datadict = onet_disk2D.data.to_datadict(data)
    u = datadict["inputs"]["u_net"]
    y = datadict["inputs"]["y_net"]
    s = datadict["s"]
    cri = [
        u.shape == (len(data["run"]), len(parameters)),
        y.shape == (len(data["r"]) * len(data["theta"]), 2),
        s.shape == (u.shape[0], y.shape[0]),
    ]
    assert all(cri)


def test_get_batch_indices():
    key = 123
    random_index_iterator = onet_disk2D.data.RandomIndexIterator(100, 10, key)
    indices_list = []
    criteria = []
    for i in range(10):
        indices = random_index_iterator.get_batch_indices()
        criteria.append(len(indices) == 10)
        indices_list.append(indices)
    indices_array = jnp.array(indices_list)

    expected = onet_disk2D.data.get_random_index_batches(
        100, 10, jax.random.PRNGKey(key)
    )
    expected = jnp.array(expected)
    criteria.append(jnp.array_equal(indices_array, expected))

    # test the 11th batch
    indices = random_index_iterator.get_batch_indices()
    # since the random key is changed, the indices of the 11th batch should be different from any of the previous 10 batches.
    equal = jnp.all(jnp.equal(indices, indices_array), axis=1)
    criteria.append(~jnp.any(equal))
    assert all(criteria)
