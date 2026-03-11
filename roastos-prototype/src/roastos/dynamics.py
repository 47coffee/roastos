from __future__ import annotations

from typing import Mapping, Any

from roastos.types import RoastState, Control

#This file implements:
#𝑋_𝑡+1 = 𝑓(𝑋_𝑡, 𝑈_𝑡,𝜃)

"""This module defines the core dynamics model for the RoastOS system, which simulates the evolution of the roast state
 based on the current state, control inputs, and optional coffee context. 
 The step_dynamics function implements a simple one-step update of the roast state, including
  bean temperature, drum energy, drying progress, Maillard progress, development progress,
   volatile loss, and structural transformation. The dynamics are designed to be stable and 
   produce plausible roast trajectories based on typical roasting behavior, while also allowing for 
   future adaptation to specific machines or coffee characteristics through the coffee_context parameter."""

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def step_dynamics(
    state: RoastState,
    control: Control,
    coffee_context: Mapping[str, Any] | None = None,
    dt_s: float = 2.0,
) -> RoastState:
    """
    One-step RoastOS toy dynamics model.

    Updates:
    - bean temperature
    - drum energy
    - drying progress
    - Maillard progress
    - development progress
    - volatile loss
    - structural transformation

    Notes:
    - This is a prototype, not a full thermodynamic model.
    - Coefficients are chosen for stable behavior and plausible roast evolution.
    - coffee_context allows later machine/bean adaptation.
    """
    coffee_context = coffee_context or {}

    # ------------------------------------------------------------------
    # 1. Read current state
    # ------------------------------------------------------------------
    Tb = state.Tb
    E_drum = state.E_drum
    p_dry = state.p_dry
    p_mai = state.p_mai
    p_dev = state.p_dev
    V_loss = state.V_loss
    S_struct = state.S_struct

    # ------------------------------------------------------------------
    # 2. Read controls (normalize % to 0..1)
    # ------------------------------------------------------------------
    gas = control.gas / 100.0
    airflow = control.airflow / 100.0
    drum_speed = control.drum_speed / 100.0

    # ------------------------------------------------------------------
    # 3. Coffee adjustments
    # Very mild for v1
    # ------------------------------------------------------------------
    density = float(coffee_context.get("density", 0.78))
    moisture = float(coffee_context.get("moisture", 0.11))

    # Denser / wetter beans heat a bit slower
    density_factor = 1.0 - 0.12 * (density - 0.78)
    moisture_factor = 1.0 - 0.25 * (moisture - 0.11)
    bean_responsiveness = density_factor * moisture_factor

    # ------------------------------------------------------------------
    # 4. Drum energy dynamics
    # Drum stores heat and responds to gas / airflow
    # ------------------------------------------------------------------
    E_drum_next = (
        0.965 * E_drum
        + 0.055 * gas
        - 0.020 * airflow
    )
    E_drum_next = _clip(E_drum_next, 0.0, 1.2)

    # ------------------------------------------------------------------
    # 5. Environment temperature proxy
    # A simple thermal driving term
    # ------------------------------------------------------------------
    T_env = 180.0 + 85.0 * gas + 18.0 * E_drum - 10.0 * airflow

    # ------------------------------------------------------------------
    # 6. Bean temperature dynamics
    # Heat transfer + direct gas effect - airflow cooling
    # ------------------------------------------------------------------
    k_env = 0.040 * bean_responsiveness
    k_gas = 1.20 * bean_responsiveness
    k_air = 0.65

    dTb = (
        k_env * (T_env - Tb)
        + k_gas * gas
        - k_air * airflow
    ) * (dt_s / 2.0)

    # Drum speed effect: slightly cleaner, slightly less aggressive heat accumulation
    dTb -= 0.12 * max(drum_speed - 0.65, 0.0)

    Tb_next = Tb + dTb
    Tb_next = _clip(Tb_next, 20.0, 260.0)

    # ------------------------------------------------------------------
    # 7. Drying progress
    # Starts building meaningfully above ~100C
    # ------------------------------------------------------------------
    dry_rate = 0.020 * max(Tb - 100.0, 0.0) / 80.0
    p_dry_next = p_dry + dry_rate * (dt_s / 2.0)
    p_dry_next = _clip(p_dry_next, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 8. Maillard progress
    # More active after drying is substantially advanced
    # ------------------------------------------------------------------
    maillard_gate = max(p_dry - 0.65, 0.0)
    maillard_temp = max(Tb - 140.0, 0.0) / 70.0
    mai_rate = 0.018 * maillard_gate * maillard_temp
    p_mai_next = p_mai + mai_rate * (dt_s / 2.0)
    p_mai_next = _clip(p_mai_next, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 9. Development progress
    # Starts near first crack region
    # ------------------------------------------------------------------
    dev_temp = max(Tb - 196.0, 0.0) / 18.0
    dev_rate = 0.022 * dev_temp
    p_dev_next = p_dev + dev_rate * (dt_s / 2.0)
    p_dev_next = _clip(p_dev_next, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 10. Volatile loss
    # Increased by heat and airflow, especially after high temp
    # ------------------------------------------------------------------
    vloss_temp = max(Tb - 185.0, 0.0) / 25.0
    vloss_rate = (
        0.010 * vloss_temp
        + 0.006 * airflow
        + 0.004 * max(gas - 0.75, 0.0)
    )
    V_loss_next = V_loss + vloss_rate * (dt_s / 2.0)
    V_loss_next = _clip(V_loss_next, 0.0, 2.0)

    # ------------------------------------------------------------------
    # 11. Structural transformation
    # Builds from Maillard + development + thermal pressure
    # ------------------------------------------------------------------
    struct_rate = (
        0.010 * (p_mai + p_dev)
        + 0.006 * max(Tb - 160.0, 0.0) / 50.0
        + 0.003 * E_drum
    )
    S_struct_next = S_struct + struct_rate * (dt_s / 2.0)
    S_struct_next = _clip(S_struct_next, 0.0, 2.0)

    return RoastState(
        Tb=Tb_next,
        E_drum=E_drum_next,
        p_dry=p_dry_next,
        p_mai=p_mai_next,
        p_dev=p_dev_next,
        V_loss=V_loss_next,
        S_struct=S_struct_next,
    )