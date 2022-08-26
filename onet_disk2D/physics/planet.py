import importlib.resources

import pandas as pd

from . import data


def read_planet_config(cfg_file: str, planet_name: str = "") -> dict:
    """

    Args:
        cfg_file:
            earth.cfg etc.
        planet_name:

    Returns:

    """
    with importlib.resources.path(data, "planet_config.csv") as f:
        planets = pd.read_csv(f, index_col=[0, 1])
    if planet_name:
        return planets.loc[(cfg_file, planet_name)].to_dict()
    else:
        planet = planets.loc[cfg_file]
        if len(planet) == 1:
            planet = planet.iloc[0]
        else:
            raise NotImplementedError(
                "Reading configs of multiple planets is not implemented."
            )
        return planet.to_dict()
