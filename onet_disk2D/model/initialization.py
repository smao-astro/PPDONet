import haiku as hk


def get_initializer(initializer: str):
    """

    References:
        https://github.com/deepmind/dm-haiku/issues/6#issuecomment-589713329
    """
    if initializer == "glorot_uniform":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
    elif initializer == "glorot_normal":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="truncated_normal"
        )
    elif initializer == "lecun_uniform":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="uniform"
        )
    elif initializer == "lecun_normal":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="truncated_normal"
        )
    elif initializer == "he_uniform":
        return hk.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="uniform"
        )
    elif initializer == "he_normal":
        return hk.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        )
    elif initializer == "sine_uniform":
        return hk.initializers.VarianceScaling(
            scale=6.0, mode="fan_in", distribution="uniform"
        )
    else:
        raise ValueError(f"initializer: {initializer} not implemented.")
