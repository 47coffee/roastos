from __future__ import annotations

from typing import Dict


def infer_phase_from_bt(bt: float, thresholds: Dict[str, float]) -> str:
    drying_end_bt = thresholds.get("drying_end_bt", 160.0)
    maillard_end_bt = thresholds.get("maillard_end_bt", 196.0)

    if bt < drying_end_bt:
        return "drying"
    if bt < maillard_end_bt:
        return "maillard"
    return "development"