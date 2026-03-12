from __future__ import annotations

from typing import Any

import numpy as np

from roastos.types import RoastState

"""This module defines the feature extraction logic for converting roast trajectories into structural roast 
features that can be used for flavor prediction and evaluation. The extract_features function takes a 
list of RoastState objects representing the roast trajectory and computes various features such as
 drying progress, Maillard progress, development progress, rate of roasting at first crack, volatile loss, 
 structural transformation, crash and flick indices, time to yellowing and first crack, and the temperature 
 rise from first crack to drop. These features are derived from the internal state variables and are 
 designed to capture key aspects of the roasting process that influence the final flavor profile of the coffee."""

INITIAL_MOISTURE_PROXY = 0.12
FIRST_CRACK_PRESSURE_THRESHOLD = 0.20
FIRST_CRACK_TEMP_MIN = 190.0
YELLOWING_DRYING_THRESHOLD = 0.80


def _find_first_index(values: list[bool]) -> int | None:
    for i, flag in enumerate(values):
        if flag:
            return i
    return None


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def extract_features(
    states: list[RoastState],
    dt_s: float = 2.0,
) -> dict[str, Any]:
    """
    Convert a roast trajectory into structural roast features.

    Expected output keys:
    - dry
    - maillard
    - dev
    - ror_fc
    - volatile_loss
    - structure
    - crash_index
    - flick_index
    - time_to_yellow_s
    - time_to_fc_s
    - dev_time_s
    - delta_bt_fc_to_drop_c
    """
    if len(states) < 3:
        raise ValueError("Need at least 3 states to extract features")

    Tb = np.array([s.Tb for s in states], dtype=float)
    RoR = np.array([s.RoR for s in states], dtype=float)
    M = np.array([s.M for s in states], dtype=float)
    P_int = np.array([s.P_int for s in states], dtype=float)
    p_mai = np.array([s.p_mai for s in states], dtype=float)
    p_dev = np.array([s.p_dev for s in states], dtype=float)
    v_loss = np.array([s.V_loss for s in states], dtype=float)
    s_struct = np.array([s.S_struct for s in states], dtype=float)

    n = len(states)
    total_time_s = (n - 1) * dt_s
    drop_idx = n - 1

    # ------------------------------------------------------------
    # Drying progress derived from moisture proxy
    # p_dry = 1 - M / M0
    # ------------------------------------------------------------
    p_dry = 1.0 - (M / INITIAL_MOISTURE_PROXY)
    p_dry = np.clip(p_dry, 0.0, 1.0)

    # ------------------------------------------------------------
    # Event detection
    # Yellowing ~ drying substantially complete
    # First crack ~ pressure threshold crossed while bean temp high enough
    # ------------------------------------------------------------
    yellow_idx = _find_first_index((p_dry >= YELLOWING_DRYING_THRESHOLD).tolist())

    fc_mask = (
        (P_int >= FIRST_CRACK_PRESSURE_THRESHOLD)
        & (Tb >= FIRST_CRACK_TEMP_MIN)
    )
    fc_idx = _find_first_index(fc_mask.tolist())

    if yellow_idx is None:
        yellow_idx = max(1, int(0.35 * (n - 1)))

    if fc_idx is None:
        # fallback: if pressure threshold not crossed, use late-roast approximation
        fc_idx = max(yellow_idx + 1, int(0.80 * (n - 1)))

    yellow_idx = min(yellow_idx, drop_idx)
    fc_idx = min(max(fc_idx, yellow_idx + 1), drop_idx)

    time_to_yellow_s = int(yellow_idx * dt_s)
    time_to_fc_s = int(fc_idx * dt_s)
    dev_time_s = int(max(drop_idx - fc_idx, 0) * dt_s)

    # ------------------------------------------------------------
    # Phase fractions
    # ------------------------------------------------------------
    pct_dry = _safe_fraction(time_to_yellow_s, total_time_s)
    pct_maillard = _safe_fraction(time_to_fc_s - time_to_yellow_s, total_time_s)
    pct_dev = _safe_fraction(dev_time_s, total_time_s)

    total_pct = pct_dry + pct_maillard + pct_dev
    if total_pct > 0:
        pct_dry /= total_pct
        pct_maillard /= total_pct
        pct_dev /= total_pct

    # ------------------------------------------------------------
    # RoR at first crack
    # RoR is now directly part of the state
    # Convert proxy from degC/s to degC/min for reporting consistency
    # ------------------------------------------------------------
    ror_fc = float(RoR[fc_idx] * 60.0)

    # ------------------------------------------------------------
    # Final structural endpoints
    # ------------------------------------------------------------
    volatile_loss = float(v_loss[-1])
    structure = float(s_struct[-1])

    # ------------------------------------------------------------
    # Crash / flick indices based on RoR after first crack
    # Crash = sharp drop after FC
    # Flick  = rebound from post-FC minimum
    # ------------------------------------------------------------
    post_fc_ror = RoR[fc_idx:] * 60.0  # convert to degC/min
    if len(post_fc_ror) >= 4:
        start_ror = float(post_fc_ror[0])
        min_ror = float(np.min(post_fc_ror))
        end_ror = float(post_fc_ror[-1])

        crash_index = max(start_ror - min_ror, 0.0) / 10.0
        flick_index = max(end_ror - min_ror, 0.0) / 10.0
    else:
        crash_index = 0.0
        flick_index = 0.0

    # ------------------------------------------------------------
    # FC-to-drop thermal rise
    # ------------------------------------------------------------
    delta_bt_fc_to_drop_c = float(Tb[-1] - Tb[fc_idx])

    return {
        "dry": float(pct_dry),
        "maillard": float(pct_maillard),
        "dev": float(pct_dev),
        "ror_fc": float(ror_fc),
        "volatile_loss": volatile_loss,
        "structure": structure,
        "crash_index": float(max(crash_index, 0.0)),
        "flick_index": float(max(flick_index, 0.0)),
        "time_to_yellow_s": time_to_yellow_s,
        "time_to_fc_s": time_to_fc_s,
        "dev_time_s": dev_time_s,
        "delta_bt_fc_to_drop_c": delta_bt_fc_to_drop_c,
    }