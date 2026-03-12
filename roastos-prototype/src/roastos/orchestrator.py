from __future__ import annotations

from pathlib import Path

from roastos.controller import RoastController
from roastos.estimator import RoastStateEstimator
from roastos.features import extract_features
from roastos.gateway.dummy_dutchmaster import DummyDutchMasterGateway
from roastos.gateway.schemas import RoastRecommendation
from roastos.mpc import RoastMPC
from roastos.state import initial_state
from roastos.types import Control

"""This module defines the main demonstration loop for the RoastOS system using a dummy gateway that simulates a Dutch Masters roasting machine.
The run_dummy_live_loop function simulates a live roasting session by iteratively reading measurement frames from the dummy gateway, updating the state estimator, optimizing control inputs with MPC, and generating recommendations using the RoastController.
The loop prints out the current measurements, estimated state, MPC objective value, and the recommended 
control adjustments along with the predicted flavor attributes at each step. This serves as a 
demonstration of how the various components of the RoastOS system interact in a live control scenario, 
and allows for testing and validation of the state estimation, MPC optimization, and control 
recommendation logic before integrating with a real roasting machine API."""

def build_target_structure(style_profile: str) -> dict:
    """
    v1 structure targets for MPC.
    Later this should come from a learned intent->structure model.
    """
    if style_profile == "filter_clarity":
        return {
            "dry": 0.40,
            "maillard": 0.44,
            "dev": 0.16,
            "volatile_loss": 0.20,
            "structure": 0.45,
            "ror_fc": 7.0,
            "Tb_end_c": 205.0,
        }

    # default espresso-ish fallback
    return {
        "dry": 0.38,
        "maillard": 0.42,
        "dev": 0.20,
        "volatile_loss": 0.28,
        "structure": 0.62,
        "ror_fc": 8.0,
        "Tb_end_c": 210.0,
    }


def run_dummy_live_loop(steps: int = 20) -> None:
    project_root = Path(__file__).resolve().parents[2]

    coffee_context = {
        "origin": "Rwanda",
        "process": "washed",
        "variety": "Bourbon",
        "density": 0.78,
        "moisture": 0.11,
        "water_activity": 0.54,
        "screen_size": 16.5,
        "altitude_m": 1850,
    }

    session_context = {
        "machine_id": "DUMMY-DM-15",
        "coffee_id": "RW-SH1",
        "operator_id": "SIMONE",
        "style_profile": "filter_clarity",
        "brew_method": "filter",
        "batch_size_kg": 6.0,
        "charge_temp_c": 205.0,
        "drop_temp_c": 205.0,
        "duration_s": 570,
        "ambient_temp_c": 21.0,
        "ambient_rh_pct": 48.0,
        "intent_clarity": 0.90,
        "intent_sweetness": 0.75,
        "intent_body": 0.35,
        "intent_bitterness": 0.15,
        "timestamp_start": "2026-03-01T10:00:00",
    }

    gateway = DummyDutchMasterGateway(
        dt_s=2.0,
        coffee_context=coffee_context,
        initial_control=Control(
            gas_pct=75.0,
            drum_pressure_pa=90.0,
            drum_speed_pct=65.0,
        ),
    )
    gateway.connect()

    estimator = RoastStateEstimator(
        initial_state=initial_state(),
        dt_s=2.0,
        coffee_context=coffee_context,
    )

    mpc = RoastMPC(horizon_steps=20, dt_s=2.0)
    controller = RoastController(model_dir=project_root / "artifacts" / "models")

    current_control = Control(
        gas_pct=75.0,
        drum_pressure_pa=90.0,
        drum_speed_pct=65.0,
    )

    target_flavor = {
        "clarity": 0.90,
        "sweetness": 0.75,
        "body": 0.35,
        "bitterness": 0.15,
    }

    target_structure = build_target_structure(session_context["style_profile"])

    print("\nRoastOS Dummy Dutch Master Live Loop")
    print("=" * 72)

    for step_idx in range(steps):
        # ------------------------------------------------------------
        # 1. Read machine frame
        # ------------------------------------------------------------
        frame = gateway.read_frame()

        # ------------------------------------------------------------
        # 2. Predict + update estimator
        # ------------------------------------------------------------
        estimator.predict(current_control)
        estimated_state = estimator.update(frame, current_control)

        # ------------------------------------------------------------
        # 3. Optimize future controls with MPC
        # ------------------------------------------------------------
        mpc_result = mpc.optimize(
            x0=estimated_state,
            current_control=current_control,
            target_structure=target_structure,
            coffee_context=coffee_context,
        )

        recommended_control = mpc_result.controls[0]

        # ------------------------------------------------------------
        # 4. Use existing controller/predictor stack to forecast flavor
        #    of the recommended sequence
        # ------------------------------------------------------------
        best_eval, _ = controller.choose_best_option(
            initial_state=estimated_state,
            candidate_control_sequences=[mpc_result.controls],
            target_flavor=target_flavor,
            session_context=session_context,
            coffee_context=coffee_context,
        )

        rec = RoastRecommendation(
            recommended_gas_pct=recommended_control.gas_pct,
            recommended_drum_pressure_pa=recommended_control.drum_pressure_pa,
            recommended_drum_speed_pct=recommended_control.drum_speed_pct,
            message=(
                f"Step {step_idx+1}: set gas to {recommended_control.gas_pct:.1f}%, "
                f"drum pressure to {recommended_control.drum_pressure_pa:.1f} Pa, "
                f"drum speed to {recommended_control.drum_speed_pct:.1f}%."
            ),
            predicted_clarity=best_eval.predicted_flavor["clarity"],
            predicted_sweetness=best_eval.predicted_flavor["sweetness"],
            predicted_body=best_eval.predicted_flavor["body"],
            predicted_bitterness=best_eval.predicted_flavor["bitterness"],
        )

        # ------------------------------------------------------------
        # 5. Print recommendation
        # ------------------------------------------------------------
        print(f"\nTime {frame.timestamp_s:6.1f}s | Machine state: {frame.machine_state}")
        print(
            f"Measured -> BT={frame.bt_c:6.2f}°C, ET={frame.et_c:6.2f}°C, "
            f"RoR={frame.ror_c_per_min:6.2f}°C/min, "
            f"Gas={frame.gas_pct:5.1f}%, Pressure={frame.drum_pressure_pa:6.1f} Pa"
        )
        print(
            f"Estimated -> Tb={estimated_state.Tb:6.2f}, RoR={estimated_state.RoR*60.0:6.2f}°C/min, "
            f"M={estimated_state.M:5.3f}, P_int={estimated_state.P_int:5.3f}, "
            f"Mai={estimated_state.p_mai:5.3f}, Dev={estimated_state.p_dev:5.3f}"
        )
        if mpc_result.success:
            print(f"MPC objective: {mpc_result.objective_value:.4f} | status={mpc_result.status}")
        else:
            print(f"MPC fallback used | status={mpc_result.status}")
        print(rec.message)
        print(
            f"Predicted flavor -> clarity={rec.predicted_clarity:.3f}, "
            f"sweetness={rec.predicted_sweetness:.3f}, "
            f"body={rec.predicted_body:.3f}, "
            f"bitterness={rec.predicted_bitterness:.3f}"
        )

        # ------------------------------------------------------------
        # 6. Simulate operator following RoastOS recommendation
        # ------------------------------------------------------------
        current_control = recommended_control
        gateway.apply_control(current_control)


if __name__ == "__main__":
    run_dummy_live_loop(steps=20)