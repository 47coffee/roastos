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
    Tb = state.Tb
    RoR = state.RoR
    E_drum = state.E_drum
    M = state.M
    P_int = state.P_int
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct

    gas = control.gas_pct / 100.0
    pressure = control.drum_pressure_pa
    drum_speed = control.drum_speed_pct / 100.0

    moisture0 = context.moisture0
    roast_progress = compute_roast_progress(state, moisture0)

    # Environment proxy
    ET_proxy = Tb + 120.0 * E_drum
    et_delta = ET_proxy - Tb

    # ------------------------------------------------------------
    # Drum energy
    # ------------------------------------------------------------
    E_drum = max(
        0.0,
        min(
            1.0,
            E_drum + (
                0.018 * gas
                - (0.010 + 0.006 * (pressure / 120.0)) * E_drum
            ) * dt_s,
        ),
    )

    # recompute environment after drum update
    ET_proxy = Tb + 60.0 * E_drum
    et_delta = ET_proxy - Tb

    # ------------------------------------------------------------
    # BT update from calibrated coefficients
    # ------------------------------------------------------------
    dTb = (
        params["intercept"]
        + params["alpha_gas"] * gas
        + params["beta_et"] * et_delta
        - params["gamma_pressure"] * pressure
        - params["delta_ror"] * (RoR * 60.0)
    )

    # scale to seconds
    Tb = Tb + dTb * dt_s

    # ------------------------------------------------------------
    # RoR update
    # ------------------------------------------------------------
    dRoR = (
        params["ror_gas_gain"] * gas
        + params["ror_et_gain"] * et_delta
        - params["ror_pressure_cooling"] * pressure
        - params["ror_progress_decay"] * roast_progress
        - 0.12 * RoR
    )

    dRoR -= 0.04 * max(drum_speed - 0.65, 0.0)

    RoR = max(-0.2, min(0.8, RoR + dRoR * dt_s))

    # ------------------------------------------------------------
    # Moisture
    # ------------------------------------------------------------
    evap = (
        params["moisture_evap_coeff"]
        * max(0.0, Tb - 140.0)
        * M
        * dt_s
    )
    M = max(0.0, M - evap)

    # ------------------------------------------------------------
    # Internal pressure
    # ------------------------------------------------------------
    pressure_build = params["pressure_build_coeff"] * max(0.0, Tb - 170.0)
    pressure_release = params["pressure_release_coeff"] * (pressure / 100.0)

    P_int = max(0.0, P_int + (pressure_build - pressure_release) * dt_s)

    # ------------------------------------------------------------
    # Maillard
    # ------------------------------------------------------------
    p_mai = max(
        0.0,
        min(1.0, p_mai + 0.0020 * max(0.0, Tb - 150.0) * dt_s),
    )

    # ------------------------------------------------------------
    # Development
    # ------------------------------------------------------------
    p_dev = max(
        0.0,
        min(1.0, p_dev + 0.0022 * max(0.0, Tb - 195.0) * dt_s),
    )

    # ------------------------------------------------------------
    # Volatile loss
    # ------------------------------------------------------------
    V_loss = max(
        0.0,
        min(1.0, V_loss + 0.0012 * max(0.0, Tb - 180.0) * dt_s),
    )

    # ------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------
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