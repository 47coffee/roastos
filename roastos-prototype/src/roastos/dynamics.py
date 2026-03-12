from __future__ import annotations

from roastos.types import RoastState, Control


def roast_dynamics(
    state: RoastState,
    control: Control,
    coffee_context: dict,
    dt_s: float,
) -> RoastState:
    """
    RoastOS Physics Model v3

    Improvements over v2
    --------------------
    - roast-progress dependent heat transfer
    - natural RoR decay
    - environment heat saturation
    - airflow / pressure cooling
    - smoother evaporation dynamics

    State units
    -----------
    Tb      °C
    RoR     °C/s
    E_drum  normalized energy
    """

    Tb = state.Tb
    RoR = state.RoR
    M = state.M
    P_int = state.P_int
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct
    E_drum = state.E_drum

    gas = control.gas_pct / 100.0
    pressure = control.drum_pressure_pa
    drum_speed = control.drum_speed_pct / 100.0

    density = coffee_context.get("density", 0.78)
    moisture0 = coffee_context.get("moisture", 0.11)

    # ---------------------------------------------------------
    # Roast progress estimate
    # ---------------------------------------------------------

    p_dry = max(0.0, min(1.0, 1.0 - M / moisture0))

    roast_progress = (
        0.45 * p_dry +
        0.40 * p_mai +
        0.15 * p_dev
    )

    roast_progress = max(0.0, min(1.0, roast_progress))

    # ---------------------------------------------------------
    # Drum energy dynamics
    # ---------------------------------------------------------

    gas_gain = 0.018 * gas
    cooling_loss = 0.010 + 0.006 * (pressure / 120.0)

    dE = gas_gain - cooling_loss * E_drum
    E_drum = max(0.0, min(1.0, E_drum + dE * dt_s))

    # ---------------------------------------------------------
    # Environment temperature proxy
    # ---------------------------------------------------------

    T_env = Tb + 170.0 * E_drum

    # ---------------------------------------------------------
    # Heat transfer efficiency
    # decreases during roast
    # ---------------------------------------------------------

    heat_eff = 1.0 - 0.65 * roast_progress
    heat_eff = max(0.20, heat_eff)

    # ---------------------------------------------------------
    # Heat transfer term
    # ---------------------------------------------------------

    deltaT = T_env - Tb

    heat_transfer = 0.020 * heat_eff * deltaT

    # ---------------------------------------------------------
    # Evaporation cooling
    # ---------------------------------------------------------

    evap_rate = 0.0035 * max(0.0, Tb - 140.0)

    evap_cooling = evap_rate * M * 6.0

    # ---------------------------------------------------------
    # Airflow cooling from drum pressure
    # ---------------------------------------------------------

    airflow_cooling = 0.004 * (pressure / 100.0) * max(0.0, Tb - 160.0)

    # ---------------------------------------------------------
    # Natural RoR decay during roast
    # ---------------------------------------------------------

    natural_decay = 0.030 * roast_progress * RoR

    # ---------------------------------------------------------
    # RoR dynamics
    # ---------------------------------------------------------

    dRoR = (
        heat_transfer
        - evap_cooling
        - airflow_cooling
        - natural_decay
        - 0.12 * RoR
    )

    RoR = RoR + dRoR * dt_s
    RoR = max(-0.5, min(2.0, RoR))

    # ---------------------------------------------------------
    # Bean temperature update
    # ---------------------------------------------------------

    Tb = Tb + RoR * dt_s

    # ---------------------------------------------------------
    # Moisture evaporation
    # ---------------------------------------------------------

    evap = evap_rate * M * dt_s
    M = max(0.0, M - evap)

    # ---------------------------------------------------------
    # Internal pressure (gas formation)
    # ---------------------------------------------------------

    pressure_build = 0.0015 * max(0.0, Tb - 170.0)

    pressure_release = 0.006 * (pressure / 100.0)

    P_int = max(0.0, P_int + (pressure_build - pressure_release) * dt_s)

    # ---------------------------------------------------------
    # Maillard progress
    # ---------------------------------------------------------

    mai_rate = 0.0020 * max(0.0, Tb - 150.0)

    p_mai = max(0.0, min(1.0, p_mai + mai_rate * dt_s))

    # ---------------------------------------------------------
    # Development progress
    # ---------------------------------------------------------

    dev_rate = 0.0022 * max(0.0, Tb - 195.0)

    p_dev = max(0.0, min(1.0, p_dev + dev_rate * dt_s))

    # ---------------------------------------------------------
    # Volatile loss
    # ---------------------------------------------------------

    vol_rate = 0.0012 * max(0.0, Tb - 180.0)

    V_loss = max(0.0, min(1.0, V_loss + vol_rate * dt_s))

    # ---------------------------------------------------------
    # Structural transformation
    # ---------------------------------------------------------

    S_struct = max(
        0.0,
        min(
            1.0,
            S_struct + (
                0.012 * p_dev
                + 0.004 * Tb / 200.0
            ) * dt_s,
        ),
    )

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
    )