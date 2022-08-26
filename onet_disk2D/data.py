import functools
import pathlib
from typing import TypedDict, List

import jax.numpy as jnp
import jax.random
import numpy as np
import xarray as xr


class InputDict(TypedDict):
    u_net: jnp.ndarray
    y_net: jnp.ndarray


class DataDict(TypedDict):
    inputs: InputDict
    s: jnp.ndarray


def load_last_frame_data(data_dir, unknown, parameter=None):
    """

    Returns:
        data: One of ('sigma', 'v_r', 'v_theta') of xr.DataArray, the data is load lazily. call `release_data` when you finish use them.
        fargo_setups: dict of fargo_setups, all values are of type string.

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


def extract_variable_parameters_name(single_data) -> List[str]:
    """

    Args:
        single_data: One of 'sigma', 'v_r', 'v_theta'

    Returns:

    """
    p_names = sorted(set(single_data.coords) - set(single_data.dims))
    return p_names


class DataIterLoader:
    """dataloader implemented from scratch, do not support multi-processing."""

    def __init__(self, data, batch_size, fixed_parameters, key=123):
        """

        Args:
            parameter_names:
                Names are in uppercase
            fargo_setups:
                Names are in lowercase
            data:
            batch_size:
            key:
        """
        data = list(data.items())
        if len(data) != 1:
            raise ValueError
        self.phys_var_type, self.data = data[0]

        self.batch_size = batch_size
        self.fixed_parameters = fixed_parameters
        self.Nu = len(self.data["run"])
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
        parameters = {
            p.lower(): data[p].values[..., None] for p in self.parameter_names
        }
        parameters.update(
            {
                p: jnp.full((len(batch_index), 1), fill_value=v)
                for p, v in self.fixed_parameters.items()
            }
        )

        data = {self.phys_var_type: to_datadict(data)}
        return parameters, data


def to_datadict(data: xr.DataArray) -> DataDict:
    """Convert the DataArray of one physics variable to DataDict.

    Notes:
        u: shape (Nu, Np)
        y: shape (Ny, 2)
        s: shape (Nu, Ny)

    Returns:

    """
    data = data.transpose("run", "r", "theta")
    parameters = list(set(data.coords) - set(data.dims))
    parameters.sort()
    u = [data[p].values for p in parameters]
    u = jnp.stack(u, axis=-1)

    r, theta = xr.broadcast(
        data.coords["r"],
        data.coords["theta"],
    )
    y = jnp.stack((r.values, theta.values), axis=-1).reshape((-1, 2))
    inputs = {"u_net": u, "y_net": y}

    s = jnp.array(data.values)
    s = s.reshape((s.shape[0], s.shape[1] * s.shape[2]))

    return {"inputs": inputs, "s": s}
