from __future__ import annotations

from roastos.types import RoastState, Control
from roastos.twin import TwinContext, step_twin
from roastos.twin_loader import load_twin_params


# ------------------------------------------------------------
# Load Digital Twin parameters
# ------------------------------------------------------------

_TWIN_PARAMS = load_twin_params()


# ------------------------------------------------------------
# Helper: normalize pressure (legacy compatibility)
# ------------------------------------------------------------

def _pressure_norm(pressure_pa: float) -> float:
    """
    Legacy helper used by estimator.

    Converts pressure to a normalized scale.
    """
    return pressure_pa / 100.0


# ------------------------------------------------------------
# Digital Twin dynamics
# ------------------------------------------------------------

def roast_dynamics(
    state: RoastState,
    control: Control,
    coffee_context: dict,
    dt_s: float = 2.0,
) -> RoastState:

    context = TwinContext(
        density=coffee_context.get("density", 0.78),
        moisture0=coffee_context.get("moisture", 0.11),
    )

    return step_twin(
        state=state,
        control=control,
        params=_TWIN_PARAMS,
        context=context,
        dt_s=dt_s,
    )


# ------------------------------------------------------------
# Compatibility wrapper expected by controller
# ------------------------------------------------------------

def step_dynamics(
    state: RoastState,
    control: Control,
    coffee_context: dict,
    dt_s: float = 2.0,
) -> RoastState:
    """
    Old interface used by RoastController.
    """
    return roast_dynamics(
        state,
        control,
        coffee_context,
        dt_s,
    )


# ------------------------------------------------------------
# Observation model (used by estimator)
# ------------------------------------------------------------

def observation_from_state(
    state: RoastState,
    control: Control,
):
    """
    Predict observable variables from latent state.

    Returns exactly:
        (BT, ET)

    This keeps backward compatibility with:
        bt, et = observation_from_state(...)
    """

    BT = state.Tb
    ET = state.Tb + 170.0 * state.E_drum

    return BT, ET
