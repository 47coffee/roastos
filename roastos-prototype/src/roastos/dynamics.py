from __future__ import annotations

import math
from typing import Mapping, Any

from roastos.types import RoastState, Control

"""This module defines the step_dynamics function, which implements the core roast dynamics model. 
The step_dynamics function takes the current roast state, control inputs, and optional coffee context parameters, 
and computes the next roast state after a time step. The dynamics model includes equations for drum energy,"""

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
    coffee_context = coffee_context or {}

    Tb = state.Tb
    RoR = state.RoR
    E_drum = state.E_drum
    M = state.M
    P_int = state.P_int
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct

    gas = control.gas / 100.0
    airflow = control.airflow / 100.0
    drum_speed = control.drum_speed / 100.0

    density = float(coffee_context.get("density", 0.78))
    moisture = float(coffee_context.get("moisture", 0.11))

    density_factor = 1.0 - 0.15 * (density - 0.78)
    moisture_factor = 1.0 - 0.35 * (moisture - 0.11)
    bean_response = _clip(density_factor * moisture_factor, 0.75, 1.25)

    # ------------------------------------------------------------
    # Drum energy
    # ------------------------------------------------------------
    E_amb = 0.35
    a_g = 0.070
    a_a = 0.035
    a_l = 0.020

    dE = (
        a_g * gas
        - a_a * airflow
        - a_l * (E_drum - E_amb)
    ) * (dt_s / 2.0)

    E_drum_next = _clip(E_drum + dE, 0.0, 1.6)

    # ------------------------------------------------------------
    # Environment temperature proxy
    # ------------------------------------------------------------
    T_base = 150.0
    b_d = 55.0
    b_g = 60.0
    b_a = 18.0

    T_env = T_base + b_d * E_drum + b_g * gas - b_a * airflow

    # ------------------------------------------------------------
    # Moisture decay
    # ------------------------------------------------------------
    T_evap = 100.0
    c_m = 0.010
    evap_gate = _sigmoid(Tb - T_evap, sharpness=0.10)
    r_evap = c_m * evap_gate * M * (0.9 + 1.2 * moisture)

    M_next = _clip(M - r_evap * (dt_s / 2.0), 0.0, 0.20)

    # ------------------------------------------------------------
    # Internal pressure
    # ------------------------------------------------------------
    T_p = 160.0
    c_p1 = 0.030
    c_p2 = 0.040
    c_p3 = 0.015

    pressure_build = c_p1 * _sigmoid(Tb - T_p, sharpness=0.12) * (M / 0.12)
    pressure_release = c_p2 * P_int + c_p3 * airflow

    P_int_next = _clip(
        P_int + (pressure_build - pressure_release) * (dt_s / 2.0),
        0.0,
        2.0,
    )

    # ------------------------------------------------------------
    # RoR dynamics
    # ------------------------------------------------------------
    k_h = 0.022 * bean_response
    k_e = 1.8
    k_r = 0.10

    dRoR = (
        k_h * (T_env - Tb)
        - k_e * M
        - k_r * RoR
    ) * (dt_s / 2.0)

    # Drum speed slightly damps aggressive acceleration
    dRoR -= 0.12 * max(drum_speed - 0.65, 0.0)

    RoR_next = _clip(RoR + dRoR, -1.0, 5.0)

    # ------------------------------------------------------------
    # Bean temperature update
    # ------------------------------------------------------------
    Tb_next = _clip(Tb + RoR_next * dt_s, 20.0, 260.0)

    # ------------------------------------------------------------
    # Maillard progress
    # ------------------------------------------------------------
    T_mai = 148.0
    c_mai = 0.012
    mai_rate = (
        c_mai
        * _sigmoid(Tb - T_mai, sharpness=0.12)
        * (1.0 - p_mai)
        * (1.0 - 0.5 * (M / 0.12))
        * math.exp(0.010 * max(Tb - 150.0, 0.0))
    )

    p_mai_next = _clip(p_mai + mai_rate * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------
    # Development progress
    # ------------------------------------------------------------
    P_fc = 0.20
    c_dev = 0.020
    dev_rate = c_dev * _sigmoid(P_int - P_fc, sharpness=8.0) * (1.0 - p_dev)

    p_dev_next = _clip(p_dev + dev_rate * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------
    # Volatile loss
    # ------------------------------------------------------------
    T_v0 = 170.0
    c_v = 0.0018
    alpha_v = 0.030
    beta_a = 0.9

    thermal_excess = max(Tb - T_v0, 0.0)
    vloss_rate = c_v * math.exp(alpha_v * thermal_excess) * (1.0 + beta_a * airflow)

    V_loss_next = _clip(V_loss + vloss_rate * (dt_s / 2.0), 0.0, 3.0)

    # ------------------------------------------------------------
    # Structural transformation
    # ------------------------------------------------------------
    T_s = 160.0
    c_s1 = 0.020
    c_s2 = 0.050
    c_s3 = 0.004

    struct_rate = (
        c_s1 * p_mai
        + c_s2 * p_dev
        + c_s3 * _sigmoid(Tb - T_s, sharpness=0.10)
    )

    S_struct_next = _clip(S_struct + struct_rate * (dt_s / 2.0), 0.0, 3.0)

    return RoastState(
        Tb=Tb_next,
        RoR=RoR_next,
        E_drum=E_drum_next,
        M=M_next,
        P_int=P_int_next,
        p_mai=p_mai_next,
        p_dev=p_dev_next,
        V_loss=V_loss_next,
        S_struct=S_struct_next,
    )