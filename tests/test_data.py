import pathlib

import numpy as np
import pytest
import xarray as xr

import onet_disk2D.data
import onet_disk2D.run.job

SINGLEP_DATAPATH = "data/singlep"
SINGLEP_FIXEDP = {
    "sigma0": 1.0,
    "nu": 1e-5,
    "sigmaslope": 0.0,
    "flaringindex": 0.0,
    "aspectratio": 0.05,
}
MULTIP_DATAPATH = "data/multip"
MULTIP_FIXEDP = {
    "sigma0": 1.0,
    "nu": 1e-5,
    "sigmaslope": 0.0,
    "flaringindex": 0.0,
}


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


class TestDataIterLoader:
    @pytest.fixture(params=[SINGLEP_DATAPATH, MULTIP_DATAPATH])
    def data_loader(self, request):
        datapath = request.param
        rawdata = onet_disk2D.data.load_last_frame_data(datapath, "log_sigma")
        data = onet_disk2D.data.DataIterLoader(rawdata, batch_size=4)
        return data

    def test_epochs(self, data_loader):
        data_iter = iter(data_loader)
        cri = []
        for i in range(10):
            data_i = next(data_iter)
            print(data_loader.key)
            for k, data in data_i.items():
                print(data["inputs"]["u_net"])

            assert True
