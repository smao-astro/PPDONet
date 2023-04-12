import functools
import pathlib
from typing import TypedDict, Mapping, Hashable

import chex
import jax.numpy as jnp
import jax.random
import numpy as np
import xarray as xr


class InputDict(TypedDict):
    u_net: chex.Array
    y_net: chex.Array


class DataDict(TypedDict):
    inputs: InputDict
    s: chex.Array


def load_last_frame_data(data_dir, unknown, parameter=None) -> dict[str, xr.DataArray]:
    """
    Args:
        data_dir: Path to the data directory.
        unknown: One of {'log_sigma', 'sigma', 'v_r', 'v_theta'}. If 'log_sigma', the data is transformed to log10(
        sigma).
        parameter: List of parameter names, all in uppercase. This is to check if the data is consistent with name of
        parameters from the commmand line input.

    """
    data_dir = pathlib.Path(data_dir)
    if unknown == "log_sigma":
        data = xr.open_dataarray(data_dir / "batch_truth_sigma.nc")
        data = np.log10(data)
    else:
        data = xr.open_dataarray(data_dir / f"batch_truth_{unknown}.nc")

    coords = list(data.coords)
    non_dims = set(data.coords) - set(data.dims)
    # only the last frame is needed
    if "t" in coords:
        raise ValueError(f"The data should only include the last frame.")
    if parameter:
        # check if parameter as expect
        if set(parameter) != non_dims:
            print(f"arg parameter: {parameter}")
            print(f"Non-dimension coordinates: {non_dims}")
            raise ValueError("parameter does not equal to non-dimension coordinates.")

    return {unknown: data}


def extract_variable_parameters_name(single_data: xr.DataArray) -> list[Hashable]:
    """

    Args:
        single_data: One of 'sigma', 'v_r', 'v_theta'

    Returns:
        p_names: sorted list of parameter names, all in uppercase.
    """
    p_names = list(set(single_data.coords) - set(single_data.dims))
    p_names.sort()
    return p_names


class DataIterLoader:
    """dataloader implemented from scratch, do not support multi-processing."""

    def __init__(self, data: Mapping[str, xr.DataArray], batch_size: int, key=123):
        data = list(data.items())
        if len(data) != 1:
            raise ValueError
        self.phys_var_type, self.data = data[0]

        self.batch_size = batch_size
        self.Nu: int = len(self.data["run"])
        self.n_batch = self.Nu // self.batch_size

        self.key = jax.random.PRNGKey(key)
        self.batch_index = None

    @functools.cached_property
    def parameter_names(self):
        return extract_variable_parameters_name(self.data)

    def init_batch_index(self):
        # call at every beginning of epochs
        # generate and shuffle index
        i = jax.random.permutation(self.key, self.Nu)
        i = jnp.array_split(i, self.n_batch)
        self.batch_index = iter(i)

    def __iter__(self):
        # determine the row (u) index for batch samples
        self.init_batch_index()
        return self

    def __next__(self):
        try:
            batch_index = next(self.batch_index)
        except StopIteration:
            print("\nEnd of a data epoch. Resampling...")
            # regenerate batch
            _, self.key = jax.random.split(self.key)
            self.init_batch_index()
            batch_index = next(self.batch_index)
        # load data
        # u shape: (Nu, 1) y shape: (Nr, Ntheta, 2) s shape (Nu, Nr, Ntheta, 1)
        data = self.data.isel(**{"run": batch_index})

        data = {self.phys_var_type: to_datadict(data)}
        return data


def extract_parameters(data: xr.DataArray) -> chex.Array:
    """

    Args:
        data:

    Returns:
        u: shape (Nu, Np). The last dimension is the parameter dimension.
    """
    parameters = extract_variable_parameters_name(data)
    u = [data[p].values for p in parameters]
    u = jnp.stack(u, axis=-1)
    return u


def to_datadict(data: xr.DataArray) -> DataDict:
    """Convert the DataArray of one physics variable to DataDict.

    Notes:
        u: shape (Nu, Np)
        y: shape (Ny, 2)
        s: shape (Nu, Ny)

    Returns:

    """
    data = data.transpose("run", "r", "theta")
    u = extract_parameters(data)

    r, theta = xr.broadcast(
        data.coords["r"],
        data.coords["theta"],
    )
    y = jnp.stack((r.values, theta.values), axis=-1).reshape((-1, 2))
    inputs = {"u_net": u, "y_net": y}

    s = jnp.array(data.values)
    s = s.reshape((s.shape[0], s.shape[1] * s.shape[2]))

    return {"inputs": inputs, "s": s}
