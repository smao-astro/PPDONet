import pathlib
import pickle

import jax
import numpy as np


def save(
    var,
    name,
    save_dir,
    leaves_file,
    struct_file,
    verbose=True,
):
    save_dir = pathlib.Path(save_dir)
    leaves = jax.tree_util.tree_leaves(var)
    # extract FlatMap structure
    struct = jax.tree_map(lambda t: 0, var)

    with open(save_dir / leaves_file, "wb") as f:
        for leave in leaves:
            np.save(f, leave, allow_pickle=False)

    with open(save_dir / struct_file, "wb") as f:
        pickle.dump(struct, f)

    if verbose:
        var_shape = jax.tree_map(lambda arr: arr.shape, var)
        print(f"\n{name} with shape :\n{var_shape}")
        print(f"Saved to {save_dir}: {leaves_file}, {struct_file}")


def save_params(
    params,
    save_dir,
    leaves_file="params.npy",
    struct_file="params_struct.pkl",
    verbose=True,
):
    save(
        var=params,
        name="Params",
        save_dir=save_dir,
        leaves_file=leaves_file,
        struct_file=struct_file,
        verbose=verbose,
    )


def save_state(
    state,
    save_dir,
    leaves_file="state.npy",
    struct_file="state_struct.pkl",
    verbose=True,
):
    save(
        var=state,
        name="State",
        save_dir=save_dir,
        leaves_file=leaves_file,
        struct_file=struct_file,
        verbose=verbose,
    )


def load(
    name,
    save_dir,
    leaves_file,
    struct_file,
    verbose=True,
):
    save_dir = pathlib.Path(save_dir)

    with open(save_dir / struct_file, "rb") as f:
        struct = pickle.load(f)
        # FlatMap

    leaves, treedef = jax.tree_util.tree_flatten(struct)
    with open(save_dir / leaves_file, "rb") as f:
        leaves = [np.load(f) for _ in leaves]

    var = jax.tree_util.tree_unflatten(treedef, leaves)

    if verbose:
        var_shape = jax.tree_map(lambda arr: arr.shape, var)
        print(f"\n{name} with shape :\n{var_shape}")
        print(f"Loaded from {save_dir}: {leaves_file}, {struct_file}")

    return var


def load_params(
    save_dir,
    leaves_file="params.npy",
    struct_file="params_struct.pkl",
    verbose=True,
):
    return load(
        name="Params",
        save_dir=save_dir,
        leaves_file=leaves_file,
        struct_file=struct_file,
        verbose=verbose,
    )


def load_state(
    save_dir,
    leaves_file="state.npy",
    struct_file="state_struct.pkl",
    verbose=True,
):
    return load(
        name="State",
        save_dir=save_dir,
        leaves_file=leaves_file,
        struct_file=struct_file,
        verbose=verbose,
    )
