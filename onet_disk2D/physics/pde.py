import jax


@jax.jit
def h(parameters, y):
    """

    Args:
        parameters: A dict of problem-specific physics parameters.
            values shape: (Nu, 1) or (1,)
        y: shape (NUM_BATCH, 2) or (2,)

    Returns:
        shape (Nu,NUM_BATCH) or (,)
    """
    r = y[..., 0]
    out = parameters["aspectratio"] * r ** parameters["flaringindex"]
    return out
