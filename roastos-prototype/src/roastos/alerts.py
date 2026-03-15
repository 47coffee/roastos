from __future__ import annotations

from dataclasses import dataclass

from roastos.types import RoastState
from roastos.gateway.schemas import RoastRecommendation
from roastos.mpc_old_casadi import MPCResult


@dataclass
class RoastAlerts:
    alert_ror_high: bool
    alert_ror_low: bool
    alert_temp_high: bool
    alert_pressure_high: bool
    alert_bitterness_risk: bool
    alert_clarity_risk: bool
    alert_mpc_fallback: bool

    def active_labels(self) -> list[str]:
        labels = []

        if self.alert_ror_high:
            labels.append("RoR high")

        if self.alert_ror_low:
            labels.append("RoR low")

        if self.alert_temp_high:
            labels.append("Temperature high")

        if self.alert_pressure_high:
            labels.append("Pressure high")

        if self.alert_bitterness_risk:
            labels.append("Bitterness risk")

        if self.alert_clarity_risk:
            labels.append("Clarity risk")

        if self.alert_mpc_fallback:
            labels.append("MPC fallback")

        return labels


def compute_alerts(
    *,
    estimated_state: RoastState,
    recommendation: RoastRecommendation,
    mpc_result: MPCResult,
) -> RoastAlerts:

    ror_c_per_min = estimated_state.RoR * 60.0

    predicted_bitterness = recommendation.predicted_bitterness or 0.0
    predicted_clarity = recommendation.predicted_clarity or 0.0

    return RoastAlerts(
        alert_ror_high=ror_c_per_min > 20.0,
        alert_ror_low=ror_c_per_min < 2.0,
        alert_temp_high=estimated_state.Tb > 225.0,
        alert_pressure_high=estimated_state.P_int > 0.25,
        alert_bitterness_risk=predicted_bitterness > 0.30,
        alert_clarity_risk=predicted_clarity < 0.78,
        alert_mpc_fallback=(not mpc_result.success),
    )