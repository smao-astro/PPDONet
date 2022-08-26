import jax.numpy as jnp
import pytest
import yaml

import onet_disk2D.physics.initial_condition as IC

SINGLEP_DATAPATH = "../data/singlep"


@pytest.fixture
def fargo_setups():
    file_path = f"{SINGLEP_DATAPATH}/fargo_setups.yml"
    with open(file_path, "r") as f:
        setups = yaml.safe_load(f)
    return {k.lower(): v for k, v in setups.items()}


@pytest.fixture
def fargo_parameters(fargo_setups):
    shape = (7, 1)
    sigma0 = jnp.full(shape, float(fargo_setups["sigma0"]))
    sigmaslope = jnp.full(shape, float(fargo_setups["sigmaslope"]))
    flaringindex = jnp.full(shape, float(fargo_setups["flaringindex"]))
    aspectratio = jnp.full(shape, float(fargo_setups["aspectratio"]))
    nu = jnp.full(shape, float(fargo_setups["nu"]))
    planetmass = jnp.full(shape, 1e-5)
    ringcenter = jnp.full(shape, 1.0)
    ringwidth = jnp.full(shape, 0.2)

    parameters = {
        "sigma0": sigma0,
        "sigmaslope": sigmaslope,
        "flaringindex": flaringindex,
        "aspectratio": aspectratio,
        "ringcenter": ringcenter,
        "ringwidth": ringwidth,
        "nu": nu,
        "planetmass": planetmass,
    }
    return parameters


@pytest.fixture
def fung_parameters(fargo_setups):
    shape = (7, 1)
    sigma0 = jnp.full(shape, float(fargo_setups["sigma0"]))
    sigmaslope = jnp.full(shape, float(fargo_setups["sigmaslope"]))
    flaringindex = jnp.full(shape, float(fargo_setups["flaringindex"]))
    aspectratio = jnp.full(shape, float(fargo_setups["aspectratio"]))
    alpha = jnp.full(shape, 0.001)
    planetmass = jnp.full(shape, 1e-5)
    ringcenter = jnp.full(shape, 1.0)
    ringwidth = jnp.full(shape, 0.2)

    parameters = {
        "sigma0": sigma0,
        "sigmaslope": sigmaslope,
        "flaringindex": flaringindex,
        "aspectratio": aspectratio,
        "ringcenter": ringcenter,
        "ringwidth": ringwidth,
        "alpha": alpha,
        "planetmass": planetmass,
    }
    return parameters


@pytest.fixture
def y():
    return jnp.linspace(0.0, 1.0, 10).reshape((5, 2))


def test_PowerlawSigmaIC(fargo_setups, fargo_parameters, y):
    ic = IC.PowerlawSigmaIC()
    ic_values = ic.func(fargo_parameters, y)

    cri = [
        ic_values.shape == (7, 5),
        jnp.array_equiv(ic_values, fargo_parameters["sigma0"]),
    ]
    assert all(cri)


def test_FungVRIC(fargo_setups, fung_parameters, y):
    ic = IC.FungVRIC()
    ic_values = ic.func(fung_parameters, y)

    cri = [
        ic_values.shape == (7, 5),
    ]

    assert all(cri)


ics = [
    IC.RingSigmaIC(),
    IC.PowerlawSigmaIC(),
    IC.KeplerianVThetaIC(omegaframe=1.0),
    IC.StaticPowerlawVThetaIC(omegaframe=1.0),
    IC.StaticRingVThetaIC(omegaframe=1.0),
    IC.FungVThetaIC(omegaframe=1.0),
    IC.StaticVRIC(),
    IC.KeplerianRingVRIC(),
]


@pytest.mark.parametrize("ic", ics)
def test_ic(ic, fargo_setups, fargo_parameters, y):
    ic_values = ic.func(fargo_parameters, y)

    cri = [ic_values.shape == (7, 5)]

    assert all(cri)


def test_get_ics(fargo_setups):
    ics = {
        "ic_sigma": IC.get_sigma_ic(fargo_setups),
        "ic_v_r": IC.get_v_r_ic(fargo_setups),
        "ic_v_theta": IC.get_v_theta_ic(fargo_setups),
    }
    print({k: type(v) for k, v in ics.items()})
    assert True
