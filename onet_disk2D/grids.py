import functools

import chex
import jax.numpy as jnp


class Grids:
    def __init__(
        self, ymin: float, ymax: float, xmin: float, xmax: float, ny: int, nx: int
    ):
        self.ymin = ymin
        self.ymax = ymax
        self.xmin = xmin
        self.xmax = xmax
        self.ny = ny
        self.nx = nx

    @functools.cached_property
    def r(self) -> chex.Array:
        """

        Returns:
            shape: (ny+1,)
        """
        return jnp.linspace(self.ymin, self.ymax, self.ny + 1)

    @functools.cached_property
    def theta(self) -> chex.Array:
        """

        Returns:
            shape: (nx+1,)
        """
        return jnp.linspace(self.xmin, self.xmax, self.nx + 1)

    @functools.cached_property
    def r_edge(self) -> chex.Array:
        """

        Returns:
            shape: (ny,)
        """
        return self.r[:-1]

    @functools.cached_property
    def r_middle(self) -> chex.Array:
        """

        Returns:
            shape: (ny,)
        """
        return (self.r[1:] + self.r[:-1]) / 2.0

    @functools.cached_property
    def theta_edge(self) -> chex.Array:
        """

        Returns:
            shape: (nx,)
        """
        return self.theta[:-1]

    @functools.cached_property
    def theta_middle(self) -> chex.Array:
        """

        Returns:
            shape: (nx,)
        """
        return (self.theta[1:] + self.theta[:-1]) / 2.0

    @functools.cached_property
    def coords_sigma(self) -> chex.Array:
        """

        Returns:
            shape: (ny, nx, 2)
        """
        r = self.r_middle
        theta = self.theta_middle
        x = jnp.stack(jnp.meshgrid(r, theta, indexing="ij"), axis=-1)
        return x

    @functools.cached_property
    def coords_v_theta(self) -> chex.Array:
        """

        Returns:
            shape: (ny, nx, 2)
        """
        r = self.r_middle
        theta = self.theta_edge
        x = jnp.stack(jnp.meshgrid(r, theta, indexing="ij"), axis=-1)
        return x

    @functools.cached_property
    def coords_v_r(self) -> chex.Array:
        """

        Returns:
            shape: (ny, nx, 2)
        """
        r = self.r_edge
        theta = self.theta_middle
        x = jnp.stack(jnp.meshgrid(r, theta, indexing="ij"), axis=-1)
        return x

    @functools.cached_property
    def r_fargo_all(self) -> dict[str, chex.Array]:
        return {
            "sigma": self.r_middle,
            "v_r": self.r_edge,
            "v_theta": self.r_middle,
        }

    @functools.cached_property
    def theta_fargo_all(self) -> dict[str, chex.Array]:
        return {
            "sigma": self.theta_middle,
            "v_r": self.theta_middle,
            "v_theta": self.theta_edge,
        }

    @functools.cached_property
    def coords_fargo_all(self) -> dict[str, chex.Array]:
        return {
            "sigma": self.coords_sigma,
            "v_r": self.coords_v_r,
            "v_theta": self.coords_v_theta,
        }
