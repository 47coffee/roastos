from __future__ import annotations

from dataclasses import dataclass

from roastos.types import RoastState, Control


@dataclass
class TwinContext:
    density: float = 0.78
    moisture0: float = 0.11


def compute_roast_progress(state: RoastState, moisture0: float) -> float:
    p_dry = max(0.0, min(1.0, 1.0 - state.M / moisture0))

    roast_progress = (
        0.45 * p_dry +
        0.40 * state.p_mai +
        0.15 * state.p_dev
    )
    return max(0.0, min(1.0, roast_progress))


def step_twin(
    state: RoastState,
    control: Control,
    params: dict,
    context: TwinContext,
    dt_s: float = 2.0,
) -> RoastState:
    """
    Stabilized RoastOS Digital Twin.

    State:
        Tb, RoR, E_drum, M, P_int, p_mai, p_dev, V_loss, S_struct, Q_bias

    Notes:
    - Q_bias is estimator-side disturbance correction
    - twin dynamics are aligned with the latest MPC
    - ET proxy and RoR dynamics are intentionally conservative for stability
    """

    Tb = state.Tb
    RoR = state.RoR
    E_drum = state.E_drum
    M = state.M
    P_int = state.P_int
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct
    Q_bias = getattr(state, "Q_bias", 0.0)

    gas = control.gas_pct / 100.0
    pressure = control.drum_pressure_pa
    drum_speed = control.drum_speed_pct / 100.0

    moisture0 = context.moisture0
    roast_progress = compute_roast_progress(state, moisture0)

    # ------------------------------------------------------------
    # Pre-update ET proxy
    # ------------------------------------------------------------
    ET_proxy = Tb + 25.0 * E_drum

    # ------------------------------------------------------------
    # Drum energy (stabilized)
    # ------------------------------------------------------------
    E_drum = E_drum + (
        0.012 * gas
        - (0.020 + 0.008 * (pressure / 120.0)) * E_drum
    ) * dt_s
    E_drum = max(0.0, min(1.0, E_drum))

    # ------------------------------------------------------------
    # Post-update ET proxy
    # ------------------------------------------------------------
    ET_proxy = Tb + 32.0 * E_drum
    et_delta = ET_proxy - Tb

    # ------------------------------------------------------------
    # BT update from blended calibrated coefficients + bias
    # ------------------------------------------------------------
    dTb = (
        params["intercept"]
        + params["alpha_gas"] * gas
        + params["beta_et"] * et_delta
        - 0.4 * params["gamma_pressure"] * pressure
    )

    Tb = Tb + (dTb + Q_bias) * dt_s

    # ---------------------------------------   ---------------------
    # Disturbance bias decay
    # ------------------------------------------------------------
    Q_bias = 0.995 * Q_bias

    # ------------------------------------------------------------
    # RoR update (aligned with MPC)
    # ------------------------------------------------------------
    thermal_drive = 0.014 * (ET_proxy - Tb)
    gas_drive = 0.040 * gas
    pressure_cooling = 0.0010 * pressure
    progress_damping = 0.14 * roast_progress

    dRoR = (
        gas_drive
        + thermal_drive
        - pressure_cooling
        - progress_damping
        - 0.55 * RoR
    )

    dRoR -= 0.02 * max(drum_speed - 0.65, 0.0)

    RoR = RoR + dRoR * dt_s

    # clamp to physical range
    RoR = max(-0.12, min(0.35, RoR))

    # guard against unrealistic strong cooling in this simplified twin
    if Tb > 120.0 and RoR < -0.05:
        RoR = -0.05

    # ------------------------------------------------------------
    # Moisture
    # ------------------------------------------------------------
    evap = (
        params["moisture_evap_coeff"]
        * max(0.0, Tb - 140.0)
        * M
        * dt_s
    )
    M = max(0.01, M - evap)

    # ------------------------------------------------------------
    # Internal pressure
    # ------------------------------------------------------------
    pressure_build = 0.70 * params["pressure_build_coeff"] * max(0.0, Tb - 178.0)
    pressure_release = 1.40 * params["pressure_release_coeff"] * (pressure / 100.0)

    P_int = max(0.0, min(0.30, P_int + (pressure_build - pressure_release) * dt_s))

    # ------------------------------------------------------------
    # Maillard (slowed)
    # ------------------------------------------------------------
    p_mai = p_mai + 0.00014 * max(0.0, Tb - 150.0) * dt_s
    p_mai = max(0.0, min(1.0, p_mai))

    # ------------------------------------------------------------
    # Development (slowed)
    # ------------------------------------------------------------
    p_dev = p_dev + 0.0008 * max(0.0, Tb - 190.0) * dt_s
    p_dev = max(0.0, min(1.0, p_dev))

    # ------------------------------------------------------------
    # Volatile loss
    # ------------------------------------------------------------
    V_loss = V_loss + 0.0012 * max(0.0, Tb - 180.0) * dt_s
    V_loss = max(0.0, min(1.0, V_loss))

    # ------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------
    S_struct = S_struct + (
        0.012 * p_dev
        + 0.004 * Tb / 200.0
    ) * dt_s
    S_struct = max(0.0, min(1.0, S_struct))

    return RoastState(
        Tb=Tb,
        RoR=RoR,
        E_drum=E_drum,
        M=M,
        P_int=P_int,
        p_mai=p_mai,
        p_dev=p_dev,
        V_loss=V_loss,
        S_struct=S_struct,
        Q_bias=Q_bias,
    )