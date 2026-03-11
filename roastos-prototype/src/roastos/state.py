from .types import RoastState


def initial_state():

    return RoastState(
        Tb=180.0,
        E_drum=0.7,
        p_dry=0.95,
        p_mai=0.75,
        p_dev=0.00,
        V_loss=0.05,
        S_struct=0.30,
    )