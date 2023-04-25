import pathlib
from typing import TypedDict, Hashable

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


def get_index_batches(total_size: int, batch_size: int) -> list[chex.Array]:
    """Split the data indices into batches.

    Args:
        total_size: Total number of data.
        batch_size: Size of each batch.

    Notes:
        If you can, make sure that total_size is divisible by batch_size.

    Returns:
        A list of index arrays, each array is the index of a batch.

    """
    indices = jnp.arange(total_size)
    return jnp.array_split(indices, total_size // batch_size)


def get_random_index_batches(
    total_size: int, batch_size: int, key: jax.random.PRNGKey
) -> list[chex.Array]:
    """Shuffle the data indices and split them into batches.

    Notes:
        If you can, make sure that total_size is divisible by batch_size.

    See `get_index_batches` for more details.
    """
    indices = jax.random.permutation(key, total_size)
    return jnp.array_split(indices, total_size // batch_size)


class RandomIndexIterator:
    def __init__(self, total_size: int, batch_size: int, key: int = 123):
        self.key = jax.random.PRNGKey(key)
        self.total_size = total_size
        self.batch_size = batch_size
        self._index_iterator = iter(
            get_random_index_batches(self.total_size, self.batch_size, self.key)
        )

    def get_batch_indices(self) -> chex.Array:
        try:
            batch_indices = next(self._index_iterator)
        except StopIteration:
            print("\nEnd of a data epoch. Resampling...")
            # update the random key
            self.key, _ = jax.random.split(self.key)
            # resample the indices
            self._index_iterator = iter(
                get_random_index_batches(self.total_size, self.batch_size, self.key)
            )
            # get the first batch
            batch_indices = next(self._index_iterator)
        return batch_indices


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
