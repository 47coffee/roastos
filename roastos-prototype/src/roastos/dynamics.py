from .types import RoastState, Control

# this implements the X_t+1 = f(X_t,U_t)

"""Defines the core dynamics function for the roasting process, modeling how the roast state evolves in response to control inputs.
This function is a simplified representation of the complex physical and chemical processes occurring during roasting, 
and is used for simulating roast trajectories and evaluating control strategies. The parameters and equations are illustrative and 
can be refined based on empirical data and domain expertise to better capture the nuances of real roasting dynamics."""

def step_dynamics(state: RoastState, control: Control):

    Tb = state.Tb
    gas = control.gas
    airflow = control.airflow

    k1 = 0.045
    k2 = 0.02
    k3 = 0.03

    Tb_next = (
        Tb
        + k1 * gas
        + k2 * (220 - Tb)
        - k3 * airflow
    )

    p_dry = min(1.0, state.p_dry + 0.01 * max(Tb - 100, 0) / 100)

    p_mai = min(1.0, state.p_mai + 0.008 * Tb / 200)

    p_dev = state.p_dev + 0.01 * max(Tb - 196, 0) / 50

    V_loss = state.V_loss + 0.005 * Tb / 200 * airflow

    S_struct = state.S_struct + 0.01 * (p_mai + p_dev)

    return RoastState(
        Tb_next,
        state.E_drum,
        p_dry,
        p_mai,
        p_dev,
        V_loss,
        S_struct,
    )