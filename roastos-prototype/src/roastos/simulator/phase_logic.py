from __future__ import annotations

from typing import Dict

from roastos.config import load_settings


def infer_phase_from_bt(bt: float, thresholds: Dict[str, float] | None = None) -> str:
    settings = load_settings()

    if thresholds is None:
        thresholds = {
            "drying_end_bt": settings.phase_thresholds.drying_end_bt,
            "maillard_end_bt": settings.phase_thresholds.maillard_end_bt,
        }

    drying_end_bt = thresholds.get("drying_end_bt", 160.0)
    maillard_end_bt = thresholds.get("maillard_end_bt", 196.0)

    if bt < drying_end_bt:
        return "drying"
    if bt < maillard_end_bt:
        return "maillard"
    return "development"