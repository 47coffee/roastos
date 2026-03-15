from __future__ import annotations

from dataclasses import dataclass

from roastos.gateway.schemas import RoastMeasurementFrame, RoastRecommendation
from roastos.mpc_old_casadi import MPCResult
from roastos.types import Control, RoastState

"""This module defines the build_recommendation function, which generates a RoastRecommendation based on the current context of the roast.
The function infers the current stage of the roast (drying, Maillard, crack approach, or development) based on the estimated internal state,
and then constructs a recommendation message that includes the recommended control adjustments, the reasoning behind the recommendation,
 and the predicted flavor attributes if the recommendation is followed."""

@dataclass
class AdvisorContext:
    current_control: Control
    recommended_control: Control
    estimated_state: RoastState
    frame: RoastMeasurementFrame
    mpc_result: MPCResult
    predicted_flavor: dict[str, float]


def _fmt_signed(value: float, unit: str = "") -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}{unit}"


def _drying_progress_from_moisture(state: RoastState, m0: float = 0.12) -> float:
    p_dry = 1.0 - (state.M / m0)
    return max(0.0, min(1.0, p_dry))


def _infer_stage(state: RoastState) -> str:
    p_dry = _drying_progress_from_moisture(state)

    if p_dry < 0.75 and state.Tb < 150.0:
        return "drying"

    if state.p_dev >= 0.03:
        return "development"

    if state.Tb >= 188 or state.P_int >= 0.18:
        return "crack_approach"

    return "maillard"


def _stage_reason_short(stage: str, state: RoastState) -> str:
    ror = state.RoR * 60.0

    if stage == "drying":
        return f"Drying incomplete, RoR {ror:.0f}"
    if stage == "maillard":
        return f"Build sweetness, RoR {ror:.0f}"
    if stage == "crack_approach":
        return "Near crack, protect clarity"
    return "Development active, limit bitterness"


def _stage_how_short(stage: str) -> str:
    if stage == "drying":
        return "Apply now, hold 4s"
    if stage == "maillard":
        return "Apply now, reassess next step"
    if stage == "crack_approach":
        return "Apply immediately, hold steady"
    return "Apply gently, avoid extra heat"


def _stage_reason_long(stage: str, state: RoastState) -> str:
    p_dry = _drying_progress_from_moisture(state)
    ror = state.RoR * 60.0

    if stage == "drying":
        return (
            f"RoastOS classifies the roast in drying because drying progress is {p_dry:.3f} "
            f"and bean temperature is {state.Tb:.1f}°C. RoR is {ror:.1f}°C/min, so the priority "
            f"is to maintain sufficient energy for moisture removal without creating excessive draft."
        )

    if stage == "maillard":
        return (
            f"RoastOS classifies the roast in Maillard because drying is substantially complete "
            f"(p_dry={p_dry:.3f}) while development remains low (p_dev={state.p_dev:.3f}). "
            f"RoR is {ror:.1f}°C/min, so the priority is sweetness development and controlled structure build."
        )

    if stage == "crack_approach":
        return (
            f"RoastOS classifies the roast as approaching crack because bean temperature is {state.Tb:.1f}°C "
            f"and internal pressure proxy is {state.P_int:.3f}, while development is still limited "
            f"(p_dev={state.p_dev:.3f}). The priority is to protect clarity, manage RoR, and limit volatile stripping."
        )

    return (
        f"RoastOS classifies the roast in development because development progress is {state.p_dev:.3f}. "
        f"Bean temperature is {state.Tb:.1f}°C and RoR is {ror:.1f}°C/min. The priority is to avoid overshooting "
        f"bitterness while preserving balance and finish."
    )


def build_recommendation(ctx: AdvisorContext) -> RoastRecommendation:
    stage = _infer_stage(ctx.estimated_state)

    curr = ctx.current_control
    rec = ctx.recommended_control
    st = ctx.estimated_state
    pf = ctx.predicted_flavor

    d_gas = rec.gas_pct - curr.gas_pct
    d_pressure = rec.drum_pressure_pa - curr.drum_pressure_pa
    d_drum = rec.drum_speed_pct - curr.drum_speed_pct

    short_what = (
        f"WHAT: Gas {_fmt_signed(d_gas, '%')}, "
        f"Pressure {_fmt_signed(d_pressure, ' Pa')}, "
        f"Drum {_fmt_signed(d_drum, '%')}"
    )
    short_why = f"WHY: {_stage_reason_short(stage, st)}"
    short_how = f"HOW: {_stage_how_short(stage)}"

    short_message = f"{short_what} | {short_why} | {short_how}"

    if ctx.mpc_result.success:
        mpc_status_text = (
            f"MPC solved successfully with objective {ctx.mpc_result.objective_value:.4f}."
        )
    else:
        mpc_status_text = (
            f"MPC fallback was used because the optimizer did not converge "
            f"({ctx.mpc_result.status})."
        )

    detailed_message = (
        f"Recommended action: change gas by {d_gas:.1f}% "
        f"(from {curr.gas_pct:.1f}% to {rec.gas_pct:.1f}%), "
        f"change drum pressure by {d_pressure:.1f} Pa "
        f"(from {curr.drum_pressure_pa:.1f} Pa to {rec.drum_pressure_pa:.1f} Pa), "
        f"and change drum speed by {d_drum:.1f}% "
        f"(from {curr.drum_speed_pct:.1f}% to {rec.drum_speed_pct:.1f}%). "
        f"{_stage_reason_long(stage, st)} "
        f"Predicted flavor if followed: clarity {pf.get('clarity', float('nan')):.3f}, "
        f"sweetness {pf.get('sweetness', float('nan')):.3f}, "
        f"body {pf.get('body', float('nan')):.3f}, "
        f"bitterness {pf.get('bitterness', float('nan')):.3f}. "
        f"{mpc_status_text}"
    )

    return RoastRecommendation(
        recommended_gas_pct=rec.gas_pct,
        recommended_drum_pressure_pa=rec.drum_pressure_pa,
        recommended_drum_speed_pct=rec.drum_speed_pct,
        message=short_message,
        detailed_message=detailed_message,
        predicted_clarity=pf.get("clarity"),
        predicted_sweetness=pf.get("sweetness"),
        predicted_body=pf.get("body"),
        predicted_bitterness=pf.get("bitterness"),
    )