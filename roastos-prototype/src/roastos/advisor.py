from __future__ import annotations

from dataclasses import dataclass

from roastos.gateway.schemas import RoastMeasurementFrame, RoastRecommendation
from roastos.mpc import MPCResult
from roastos.types import Control, RoastState

"""This module defines the AdvisorContext dataclass and the build_recommendation function, which together
form the core of the advisory component of the RoastOS system. The AdvisorContext class encapsulates all the 
relevant information needed to generate a control recommendation, including the current control inputs, 
the recommended control adjustments, the estimated state of the roast, the latest measurement frame from 
the machine, the result of the MPC optimization, and the predicted flavor attributes. The build_recommendation 
function takes this context as input and constructs a RoastRecommendation object that includes not only the 
recommended control adjustments but also a detailed message explaining the reasoning behind the recommendation, 
the predicted flavor outcomes if the recommendation is followed, and the status of the MPC optimization. 
This function serves as a key part of how RoastOS communicates actionable insights to users or external 
interfaces based on real-time data and model predictions during the roasting process."""

@dataclass
class AdvisorContext:
    current_control: Control
    recommended_control: Control
    estimated_state: RoastState
    frame: RoastMeasurementFrame
    mpc_result: MPCResult
    predicted_flavor: dict[str, float]


def _fmt_delta(new_value: float, old_value: float, unit: str = "") -> str:
    delta = new_value - old_value
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}{unit}"


def _infer_stage(state: RoastState) -> str:
    p_dry = max(0.0, min(1.0, 1.0 - (state.M / 0.12)))

    if p_dry < 0.80:
        return "drying"

    if state.P_int < 0.20 and state.p_dev < 0.01:
        return "maillard"

    if state.P_int >= 0.20 and state.p_dev < 0.03:
        return "crack_approach"

    return "development"


def build_recommendation(ctx: AdvisorContext) -> RoastRecommendation:
    stage = _infer_stage(ctx.estimated_state)

    curr = ctx.current_control
    rec = ctx.recommended_control
    st = ctx.estimated_state
    pf = ctx.predicted_flavor

    gas_delta = rec.gas_pct - curr.gas_pct
    pressure_delta = rec.drum_pressure_pa - curr.drum_pressure_pa
    drum_delta = rec.drum_speed_pct - curr.drum_speed_pct

    if stage == "drying":
        reason = (
            f"Drying stage. Moisture remains elevated at {st.M:.3f}. "
            f"Keep enough energy to maintain momentum while avoiding excessive draft."
        )
    elif stage == "maillard":
        reason = (
            f"Maillard stage. RoR is {st.RoR*60.0:.1f}°C/min and pressure is {st.P_int:.3f}. "
            f"Manage heat and draft to build sweetness without pushing body too aggressively."
        )
    elif stage == "crack_approach":
        reason = (
            f"Approaching crack regime. Internal pressure proxy is {st.P_int:.3f}. "
            f"Protect clarity by controlling RoR and using draft to limit volatile loss."
        )
    else:
        reason = (
            f"Development stage. Development progress is {st.p_dev:.3f}. "
            f"Avoid overshooting bitterness while preserving balance and finish."
        )

    action_message = (
        f"Set gas to {rec.gas_pct:.1f}% "
        f"({_fmt_delta(rec.gas_pct, curr.gas_pct, '%')} vs current), "
        f"drum pressure to {rec.drum_pressure_pa:.1f} Pa "
        f"({_fmt_delta(rec.drum_pressure_pa, curr.drum_pressure_pa, ' Pa')}), "
        f"drum speed to {rec.drum_speed_pct:.1f}% "
        f"({_fmt_delta(rec.drum_speed_pct, curr.drum_speed_pct, '%')})."
    )

    performance_hint = (
        f"Predicted flavor if followed: clarity {pf['clarity']:.3f}, "
        f"sweetness {pf['sweetness']:.3f}, body {pf['body']:.3f}, "
        f"bitterness {pf['bitterness']:.3f}."
    )

    if ctx.mpc_result.success:
        status_hint = f"MPC solved successfully (objective {ctx.mpc_result.objective_value:.4f})."
    else:
        status_hint = f"MPC fallback used ({ctx.mpc_result.status})."

    full_message = f"{action_message} {reason} {performance_hint} {status_hint}"

    return RoastRecommendation(
        recommended_gas_pct=rec.gas_pct,
        recommended_drum_pressure_pa=rec.drum_pressure_pa,
        recommended_drum_speed_pct=rec.drum_speed_pct,
        message=full_message,
        predicted_clarity=pf.get("clarity"),
        predicted_sweetness=pf.get("sweetness"),
        predicted_body=pf.get("body"),
        predicted_bitterness=pf.get("bitterness"),
    )