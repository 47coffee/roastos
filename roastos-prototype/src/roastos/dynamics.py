from __future__ import annotations

import math
from typing import Mapping, Any

from roastos.types import RoastState, Control


MIN_PRESSURE_PA = 50.0
MAX_PRESSURE_PA = 150.0

"""This module defines the dynamics model for simulating the roast process. The step_dynamics function 
takes the current state of the roast, the control inputs (gas, airflow, drum speed), and optional 
coffee context parameters (density and moisture) to compute the next state of the roast after a time step. 
The model includes equations for updating the drum energy, environment temperature proxy, moisture decay, internal pressure, 
rate of rise (RoR), bean temperature, Maillard progress, development progress, volatile loss, and structural 
transformation based on the current state and control inputs. The dynamics are designed to capture key interactions 
between these variables in a simplified way that allows for simulating future trajectories under different control plans."""

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(x: float, sharpness: float = 0.12) -> float:
    return 1.0 / (1.0 + math.exp(-sharpness * x))


def _pressure_norm(pressure_pa: float) -> float:
    """
    Normalize drum pressure from physical range [50, 150] Pa to [0, 1].
    """
    return _clip((pressure_pa - MIN_PRESSURE_PA) / (MAX_PRESSURE_PA - MIN_PRESSURE_PA), 0.0, 1.0)


def observation_from_state(
    state: RoastState,
    control: Control,
) -> tuple[float, float]:
    """
    Very simple observation model:
    returns (BT_sensor, ET_sensor)
    """
    gas = control.gas_pct / 100.0
    pnorm = _pressure_norm(control.drum_pressure_pa)

    bt = state.Tb
    et = 170.0 + 40.0 * state.E_drum + 35.0 * gas - 10.0 * pnorm
    return bt, et


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

    gas = control.gas_pct / 100.0
    pnorm = _pressure_norm(control.drum_pressure_pa)
    drum_speed = control.drum_speed_pct / 100.0

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
    a_p = 0.035
    a_l = 0.020

    dE = (
        a_g * gas
        - a_p * pnorm
        - a_l * (E_drum - E_amb)
    ) * (dt_s / 2.0)

    E_drum_next = _clip(E_drum + dE, 0.0, 1.6)

    # ------------------------------------------------------------
    # Environment temperature proxy
    # more pressure => more draft => stronger heat removal
    # ------------------------------------------------------------
    T_base = 150.0
    b_d = 55.0
    b_g = 60.0
    b_p = 20.0

    T_env = T_base + b_d * E_drum + b_g * gas - b_p * pnorm

    # ------------------------------------------------------------
    # Moisture decay
    # now drying is more sensitive to RoR / energy flow
    # ------------------------------------------------------------
    T_evap = 100.0
    c_m = 0.010
    evap_gate = _sigmoid(Tb - T_evap, sharpness=0.10)
    r_evap = (
        c_m
        * evap_gate
        * M
        * (0.9 + 1.2 * moisture)
        * (1.0 + 0.25 * max(RoR, 0.0))
    )

    M_next = _clip(M - r_evap * (dt_s / 2.0), 0.0, 0.20)

    # ------------------------------------------------------------
    # Internal pressure
    # more temperature + more remaining moisture => more pressure
    # pressure is relieved by stronger draft
    # ------------------------------------------------------------
    T_p = 160.0
    c_p1 = 0.030
    c_p2 = 0.040
    c_p3 = 0.020

    pressure_build = c_p1 * _sigmoid(Tb - T_p, sharpness=0.12) * (M / 0.12)
    pressure_release = c_p2 * P_int + c_p3 * pnorm

    P_int_next = _clip(
        P_int + (pressure_build - pressure_release) * (dt_s / 2.0),
        0.0,
        2.0,
    )

    # ------------------------------------------------------------
    # RoR heat-balance dynamics
    # ------------------------------------------------------------
    k_h = 0.022 * bean_response
    k_e = 1.8
    k_r = 0.10

    dRoR = (
        k_h * (T_env - Tb)
        - k_e * M
        - k_r * RoR
    ) * (dt_s / 2.0)

    # faster drum speed slightly damps aggressive acceleration
    dRoR -= 0.12 * max(drum_speed - 0.65, 0.0)

    RoR_next = _clip(RoR + dRoR, -1.0, 5.0)

    # ------------------------------------------------------------
    # Bean temperature update
    # ------------------------------------------------------------
    Tb_next = _clip(Tb + RoR_next * dt_s, 20.0, 260.0)

    # ------------------------------------------------------------
    # Maillard progress
    # stronger temperature sensitivity
    # ------------------------------------------------------------
    T_mai = 148.0
    c_mai = 0.010
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
    # pressure-driven crack regime
    # ------------------------------------------------------------
    P_fc = 0.20
    c_dev = 0.020
    dev_rate = c_dev * _sigmoid(P_int - P_fc, sharpness=15.0) * (1.0 - p_dev)

    p_dev_next = _clip(p_dev + dev_rate * (dt_s / 2.0), 0.0, 1.0)

    # ------------------------------------------------------------
    # Volatile loss
    # more pressure/draft => more stripping
    # ------------------------------------------------------------
    T_v0 = 170.0
    c_v = 0.0018
    alpha_v = 0.030
    beta_p = 0.9

    thermal_excess = max(Tb - T_v0, 0.0)
    vloss_rate = c_v * math.exp(alpha_v * thermal_excess) * (1.0 + beta_p * pnorm)

    V_loss_next = _clip(V_loss + vloss_rate * (dt_s / 2.0), 0.0, 3.0)

    # ------------------------------------------------------------
    # Structural transformation
    # ------------------------------------------------------------
    T_s = 160.0
    c_s1 = 0.040
    c_s2 = 0.090
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