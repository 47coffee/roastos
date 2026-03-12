from __future__ import annotations

import math
from typing import Mapping, Any

from roastos.types import RoastState, Control

"""
This module defines the physical dynamics model for the coffee roasting process."""

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(x: float, sharpness: float = 0.12) -> float:
    return 1.0 / (1.0 + math.exp(-sharpness * x))


def step_dynamics(
    state: RoastState,
    control: Control,
    coffee_context: Mapping[str, Any] | None = None,
    dt_s: float = 2.0,
) -> RoastState:
    """
    Improved RoastOS physical dynamics model.

    State:
        Tb        bean temperature
        E_drum    stored drum energy
        p_dry     drying progress
        p_mai     Maillard progress
        p_dev     development progress
        V_loss    volatile loss
        S_struct  structural transformation

    Control inputs:
        gas, airflow, drum_speed in percent
    """
    coffee_context = coffee_context or {}

    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    Tb = state.Tb
    E_drum = state.E_drum
    p_dry = state.p_dry
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct

    # ------------------------------------------------------------------
    # 2. Normalize controls to 0..1
    # ------------------------------------------------------------------
    gas = control.gas / 100.0
    airflow = control.airflow / 100.0
    drum_speed = control.drum_speed / 100.0

    # ------------------------------------------------------------------
    # 3. Coffee context adjustments
    # ------------------------------------------------------------------
    density = float(coffee_context.get("density", 0.78))
    moisture = float(coffee_context.get("moisture", 0.11))

    # Dense / wetter coffees respond more slowly
    density_factor = 1.0 - 0.18 * (density - 0.78)
    moisture_factor = 1.0 - 0.45 * (moisture - 0.11)
    bean_response = density_factor * moisture_factor

    # Clamp to reasonable range
    bean_response = _clip(bean_response, 0.75, 1.25)

    # ------------------------------------------------------------------
    # 4. Drum energy dynamics
    # ------------------------------------------------------------------
    E_amb = 0.35
    a_g = 0.070
    a_a = 0.035
    a_l = 0.020

    dE_drum = (
        a_g * gas
        - a_a * airflow
        - a_l * (E_drum - E_amb)
    ) * (dt_s / 2.0)

    E_drum_next = _clip(E_drum + dE_drum, 0.0, 1.6)

    # ------------------------------------------------------------------
    # 5. Roasting environment temperature proxy
    # ------------------------------------------------------------------
    T_base = 150.0
    b_d = 52.0
    b_g = 58.0
    b_a = 18.0

    T_env = T_base + b_d * E_drum + b_g * gas - b_a * airflow

    # ------------------------------------------------------------------
    # 6. Drying / evaporation heat sink
    # Strongest while p_dry is incomplete and Tb > evaporation threshold
    # ------------------------------------------------------------------
    T_evap = 100.0
    evap_gate = max(Tb - T_evap, 0.0) / 40.0
    phi_dry = evap_gate * (1.0 - p_dry) * (0.8 + 1.2 * moisture)

    # ------------------------------------------------------------------
    # 7. Bean temperature dynamics
    # ------------------------------------------------------------------
    k_h = 0.060 * bean_response
    k_evap = 2.8

    dTb = (
        k_h * (T_env - Tb)
        - k_evap * phi_dry
    ) * (dt_s / 2.0)

    # Faster drum speed slightly reduces aggressive heat pickup
    dTb -= 0.18 * max(drum_speed - 0.65, 0.0)

    Tb_next = _clip(Tb + dTb, 20.0, 260.0)

    # ------------------------------------------------------------------
    # 8. Drying progress
    # ------------------------------------------------------------------
    c_dry = 0.020
    r_dry = (
        c_dry
        * max(Tb - T_evap, 0.0)
        / 35.0
        * (1.0 - p_dry)
        * (0.85 + 1.4 * moisture)
    )
    p_dry_next = _clip(p_dry + r_dry * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 9. Maillard progress
    # Activated after drying and after ~145-150C
    # ------------------------------------------------------------------
    T_mai = 148.0
    c_mai = 0.018
    mai_gate = _sigmoid(Tb - T_mai, sharpness=0.12)
    r_mai = c_mai * mai_gate * p_dry * (1.0 - p_mai)
    p_mai_next = _clip(p_mai + r_mai * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 10. Development progress
    # Activated near first crack
    # ------------------------------------------------------------------
    T_fc = 196.0
    c_dev = 0.028
    dev_gate = _sigmoid(Tb - T_fc, sharpness=0.18)
    r_dev = c_dev * dev_gate * (1.0 - p_dev)
    p_dev_next = _clip(p_dev + r_dev * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 11. Volatile loss
    # Nonlinear increase with temperature + airflow stripping
    # ------------------------------------------------------------------
    T_v0 = 170.0
    c_v = 0.0018
    alpha_v = 0.030
    beta_a = 0.9

    thermal_excess = max(Tb - T_v0, 0.0)
    r_vloss = c_v * math.exp(alpha_v * thermal_excess) * (1.0 + beta_a * airflow)
    V_loss_next = _clip(V_loss + r_vloss * (dt_s / 2.0), 0.0, 3.0)

    # ------------------------------------------------------------------
    # 12. Structural transformation
    # Builds from Maillard + development + thermal load
    # ------------------------------------------------------------------
    T_s = 160.0
    c_s1 = 0.010
    c_s2 = 0.022
    c_s3 = 0.0035

    r_struct = (
        c_s1 * p_mai
        + c_s2 * p_dev
        + c_s3 * max(Tb - T_s, 0.0) / 30.0
    )
    S_struct_next = _clip(S_struct + r_struct * (dt_s / 2.0), 0.0, 3.0)

    return RoastState(
        Tb=Tb_next,
        E_drum=E_drum_next,
        p_dry=p_dry_next,
        p_mai=p_mai_next,
        p_dev=p_dev_next,
        V_loss=V_loss_next,
        S_struct=S_struct_next,
    )