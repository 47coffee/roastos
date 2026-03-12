import numpy as np

"""This module defines the observation model for mapping the internal state of the roast to observable features."""

def observation_model(x):

    Tb = x[0]
    E_drum = x[1]

    BT = Tb
    ET = Tb + 0.5*(E_drum - Tb)

    return np.array([BT, ET])