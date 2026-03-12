from __future__ import annotations

from typing import Any

import numpy as np

from roastos.types import RoastState

"""This module defines the feature extraction logic for converting roast trajectories into structural roast features."""

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
    """
    if len(states) < 3:
        raise ValueError("Need at least 3 states to extract features")

    Tb = np.array([s.Tb for s in states], dtype=float)
    p_dry = np.array([s.p_dry for s in states], dtype=float)
    p_mai = np.array([s.p_mai for s in states], dtype=float)
    p_dev = np.array([s.p_dev for s in states], dtype=float)
    v_loss = np.array([s.V_loss for s in states], dtype=float)
    s_struct = np.array([s.S_struct for s in states], dtype=float)

    n = len(states)
    total_time_s = (n - 1) * dt_s
    dTb_dt = np.gradient(Tb, dt_s) * 60.0  # C/min

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------
    yellow_idx = _find_first_index((p_dry >= 0.80).tolist())
    fc_idx = _find_first_index((Tb >= 196.0).tolist())
    drop_idx = n - 1

    if yellow_idx is None:
        yellow_idx = max(1, int(0.35 * (n - 1)))
    if fc_idx is None:
        fc_idx = max(yellow_idx + 1, int(0.80 * (n - 1)))

    yellow_idx = min(yellow_idx, drop_idx)
    fc_idx = min(max(fc_idx, yellow_idx + 1), drop_idx)

    time_to_yellow_s = int(yellow_idx * dt_s)
    time_to_fc_s = int(fc_idx * dt_s)
    dev_time_s = int(max(drop_idx - fc_idx, 0) * dt_s)

    # ------------------------------------------------------------------
    # Phase fractions
    # ------------------------------------------------------------------
    pct_dry = _safe_fraction(time_to_yellow_s, total_time_s)
    pct_maillard = _safe_fraction(time_to_fc_s - time_to_yellow_s, total_time_s)
    pct_dev = _safe_fraction(dev_time_s, total_time_s)

    total_pct = pct_dry + pct_maillard + pct_dev
    if total_pct > 0:
        pct_dry /= total_pct
        pct_maillard /= total_pct
        pct_dev /= total_pct

    # ------------------------------------------------------------------
    # RoR at first crack
    # ------------------------------------------------------------------
    ror_fc = float(dTb_dt[fc_idx])

    # ------------------------------------------------------------------
    # Volatile loss / structure endpoints
    # ------------------------------------------------------------------
    volatile_loss = float(v_loss[-1])
    structure = float(s_struct[-1])

    # ------------------------------------------------------------------
    # Crash / flick indices
    # Crash = sharp drop after FC
    # Flick  = late recovery after the post-FC minimum
    # ------------------------------------------------------------------
    post_fc_ror = dTb_dt[fc_idx:]
    if len(post_fc_ror) >= 4:
        start_ror = float(post_fc_ror[0])
        min_ror = float(np.min(post_fc_ror))
        end_ror = float(post_fc_ror[-1])

        crash_index = max(start_ror - min_ror, 0.0) / 10.0
        flick_index = max(end_ror - min_ror, 0.0) / 10.0
    else:
        crash_index = 0.0
        flick_index = 0.0

    # ------------------------------------------------------------------
    # FC-to-drop thermal rise
    # ------------------------------------------------------------------
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