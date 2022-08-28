import pathlib
import shutil

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
