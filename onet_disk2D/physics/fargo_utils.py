import numpy as np


def get_frame_angular_velocity(frame, omegaframe, planet_distance):
    if frame == "F":
        return omegaframe
    elif frame in ["C", "G"]:
        if np.isclose(planet_distance, 0.0):
            raise ValueError(
                f"planet_distance = {planet_distance} is close to zero. Can not set rotating frame."
            )
        return planet_distance**-1.5
    else:
        raise KeyError
