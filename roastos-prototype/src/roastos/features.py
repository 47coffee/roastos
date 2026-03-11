import numpy as np

# this is the x = q(X_trajectory) , basically is the transformation
# from trajectory to structure. the structure will then be used to determine
# the falvour

"""This module defines the feature extraction process, transforming raw roast trajectories into structured features 
that can be used for flavor prediction. The `extract_features` function takes a sequence of roast states and 
computes key features such as dryness, Maillard reaction progress, development level, rate of rise at first crack,
 volatile loss, and structural integrity. These features serve as the input to the flavor prediction model, encapsulating the essential characteristics of the roast that influence flavor outcomes."""


def extract_features(states):

    Tb = [s.Tb for s in states]
    p_dry = states[-1].p_dry
    p_mai = states[-1].p_mai
    p_dev = states[-1].p_dev
    V_loss = states[-1].V_loss
    S_struct = states[-1].S_struct

    ror_fc = np.gradient(Tb)[-1]

    return {
        "dry": p_dry,
        "maillard": p_mai,
        "dev": p_dev,
        "ror_fc": ror_fc,
        "volatile_loss": V_loss,
        "structure": S_struct,
    }