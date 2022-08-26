import pytest
import jax.numpy as jnp
import onet_disk2D.grids

ymin = 0.4
ymax = 2.5
xmin = -3
xmax = 3
ny = 10
nx = 6


@pytest.fixture
def grids():
    return onet_disk2D.grids.Grids(ymin, ymax, xmin, xmax, ny, nx)


def test_coords_dens(grids):
    coords = grids.coords_sigma
    cri = [
        jnp.isclose(
            jnp.min(coords[..., 0]),
            grids.ymin + (grids.ymax - grids.ymin) / grids.ny / 2.0,
        )
    ]
    assert all(cri)
