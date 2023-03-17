import jax.numpy as jnp
import numpy as np
import pytest

import onet_disk2D.physics.initial_condition as IC
import onet_disk2D.run

SINGLEP_DATAPATH = "../data/singlep"


@pytest.fixture
def fargo_setups():
    file_path = f"{SINGLEP_DATAPATH}/fargo_setups.yml"
    setups, _ = onet_disk2D.run.load_fargo_setups(file_path)
    return setups


@pytest.fixture
def fung_parameters(fargo_setups):
    parameters = fargo_setups.copy()
    p_inputs = {"planetmass": (0,)}
    for key in p_inputs:
        if key in parameters:
            raise KeyError(f"{key} is already in parameters.")
    parameters.update(p_inputs)
    return parameters


@pytest.fixture
def y():
    """a 2D grid of (r, theta)"""
    r_min = 0.4
    r_max = 2.5
    Nr = 5
    Ntheta = 3
    r = jnp.linspace(r_min, r_max, Nr)
    theta_min = -jnp.pi
    theta_max = jnp.pi
    theta = jnp.linspace(theta_min, theta_max, Ntheta)
    r, theta = jnp.meshgrid(r, theta)
    y = jnp.stack([r.flatten(), theta.flatten()], axis=-1)
    return y


def test_RingSigmaIC(y):
    u = np.stack(
        [
            np.linspace(0.5, 1.5, 7),  # sigma0
            np.linspace(0.5, 1.0, 7),  # sigmaslope
        ],
        axis=-1,
    )
    ringcenter = 1.0
    ringwidth = 0.1
    ic = IC.RingSigmaIC(
        sigma0=(0,),
        ringcenter=ringcenter,
        ringwidth=ringwidth,
    )
    ic_values = ic.func(u, y)

    r = y[..., 0]
    truth = u[..., 0, None] * (
        1.0
        + jnp.exp(-((r - ringcenter) ** 2) / (2 * ringwidth**2))
        / (jnp.sqrt(2 * jnp.pi) * ringwidth)
    )

    cri = [
        ic_values.shape == (7, 15),
        jnp.array_equiv(ic_values, truth),
    ]
    assert all(cri)


def test_PowerlawSigmaIC(y):
    u = np.stack(
        [
            np.linspace(0.5, 1.5, 7),  # sigma0
            np.linspace(0.5, 1.0, 7),  # sigmaslope
        ],
        axis=-1,
    )
    ic = IC.PowerlawSigmaIC(sigma0=(0,), sigmaslope=(1,))
    ic_values = ic.func(u, y)

    r = y[..., 0]
    truth = u[..., 0, None] * r ** -u[..., 1, None]

    cri = [
        ic_values.shape == (7, 15),
        jnp.array_equiv(ic_values, truth),
    ]
    assert all(cri)


def test_KeplerianVThetaIC(y):
    u = np.stack(
        [
            np.linspace(0.5, 1.5, 7),  # sigma0
            np.linspace(0.5, 1.0, 7),  # sigmaslope
        ],
        axis=-1,
    )
    ic = IC.KeplerianVThetaIC(omegaframe=1.0)
    ic_values = ic.func(u, y)

    cri = [
        ic_values.shape == (7, 15),
    ]
    assert all(cri)


def test_StaticVRIC(y):
    u = np.stack(
        [
            np.linspace(0.5, 1.5, 7),  # sigma0
            np.linspace(0.5, 1.0, 7),  # sigmaslope
        ],
        axis=-1,
    )
    ic = IC.StaticVRIC()
    ic_values = ic.func(u, y)

    cri = [
        ic_values.shape == (7, 15),
    ]
    assert all(cri)


#
#
def test_FungVRIC(y):
    u = np.stack(
        [
            np.logspace(-4, -2, 7),  # alpha
            np.linspace(0.5, 1.0, 7),  # aspectratio
        ],
        axis=-1,
    )
    ic = IC.FungVRIC(alpha=(0,), aspectratio=(1,), flaringindex=0.0, sigmaslope=0.5)
    ic_values = ic.func(u, y)

    cri = [
        ic_values.shape == (7, 15),
    ]

    assert all(cri)


def test_get_ics(fung_parameters):
    ics = {
        "ic_sigma": IC.get_sigma_ic(fung_parameters["densityinitial"], fung_parameters),
        "ic_v_r": IC.get_v_r_ic(fung_parameters["vyinitial"], fung_parameters),
        "ic_v_theta": IC.get_v_theta_ic(fung_parameters["vxinitial"], fung_parameters),
    }
    assert True
