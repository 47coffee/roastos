from __future__ import annotations

from pathlib import Path

from roastos.advisor import AdvisorContext, build_recommendation
from roastos.alerts import compute_alerts
from roastos.controller import RoastController
from roastos.estimator import RoastStateEstimator
from roastos.filter import RoRFilter
from roastos.gateway.dummy_dutchmaster import DummyDutchMasterGateway
from roastos.logger import RoastRuntimeLogger
from roastos.mpc import RoastMPC
from roastos.state import initial_state
from roastos.types import Control


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

    return {
        "dry": 0.38,
        "maillard": 0.42,
        "dev": 0.20,
        "volatile_loss": 0.28,
        "structure": 0.62,
        "ror_fc": 8.0,
        "Tb_end_c": 210.0,
    }


def run_dummy_live_loop(steps: int = 20, dt_s: float = 2.0) -> None:
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

    current_control = Control(
        gas_pct=75.0,
        drum_pressure_pa=90.0,
        drum_speed_pct=65.0,
    )

    gateway = DummyDutchMasterGateway(
        dt_s=dt_s,
        coffee_context=coffee_context,
        initial_control=current_control,
    )
    gateway.connect()

    estimator = RoastStateEstimator(initial_state=initial_state())
    ror_filter = RoRFilter(alpha=0.25)

    mpc = RoastMPC(horizon_steps=12, dt_s=dt_s, n_blocks=2)
    controller = RoastController(model_dir=project_root / "artifacts" / "models")
    logger = RoastRuntimeLogger(project_root / "artifacts" / "runtime_log.csv")

    target_flavor = {
        "clarity": 0.90,
        "sweetness": 0.75,
        "body": 0.35,
        "bitterness": 0.15,
        "acidity_quality": 0.65,
    }
    target_structure = build_target_structure(session_context["style_profile"])

    print("\nRoastOS Dummy Dutch Master Live Loop")
    print("=" * 72)

    for _step_idx in range(steps):
        # ------------------------------------------------------------
        # 1. Read machine frame
        # ------------------------------------------------------------
        frame = gateway.read_frame()

        # ------------------------------------------------------------
        # 2. Filter measured RoR for display / future use
        # ------------------------------------------------------------
        filtered_ror_c_per_min = ror_filter.update(frame.ror_c_per_min)

        # ------------------------------------------------------------
        # 3. EKF predict + update
        # ------------------------------------------------------------
        estimator.predict(
            control=current_control,
            coffee_context=coffee_context,
            dt_s=dt_s,
        )
        estimated_state = estimator.update(
            bt_meas=frame.bt_c,
            et_meas=frame.et_c,
            control=current_control,
        )

        # Optional: lightly blend measured filtered RoR into estimated RoR
        # without overriding the EKF completely.
        estimated_state.RoR = 0.8 * estimated_state.RoR + 0.2 * (filtered_ror_c_per_min / 60.0)

        # ------------------------------------------------------------
        # 4. Solve MPC on estimated state
        # ------------------------------------------------------------
        
        mpc_result = mpc.optimize(
            x0=estimated_state,
            current_control=current_control,
            target_structure=target_structure,
            coffee_context=coffee_context,
        )

        raw_control = mpc_result.controls[0]

        # ------------------------------------------------------------
        # Actuator saturation (prevents floating point overflow)
        # ------------------------------------------------------------
        gas = max(0.0, min(100.0, raw_control.gas_pct))
        pressure = max(50.0, min(120.0, raw_control.drum_pressure_pa))
        drum = max(55.0, min(75.0, raw_control.drum_speed_pct))

        recommended_control = Control(
            gas_pct=gas,
            drum_pressure_pa=pressure,
            drum_speed_pct=drum,
        )

        # ------------------------------------------------------------
        # 5. Forecast flavor for the recommended horizon
        # ------------------------------------------------------------
        best_eval, _ = controller.choose_best_option(
            initial_state=estimated_state,
            candidate_control_sequences=[mpc_result.controls],
            target_flavor=target_flavor,
            session_context=session_context,
            coffee_context=coffee_context,
        )

        # ------------------------------------------------------------
        # 6. Build operator recommendation
        # ------------------------------------------------------------
        recommendation = build_recommendation(
            AdvisorContext(
                current_control=current_control,
                recommended_control=recommended_control,
                estimated_state=estimated_state,
                frame=frame,
                mpc_result=mpc_result,
                predicted_flavor=best_eval.predicted_flavor,
            )
        )

        # ------------------------------------------------------------
        # 7. Compute alerts
        # ------------------------------------------------------------
        alerts = compute_alerts(
            estimated_state=estimated_state,
            recommendation=recommendation,
            mpc_result=mpc_result,
        )

        # ------------------------------------------------------------
        # 8. Print live status
        # ------------------------------------------------------------
        print(f"\nTime {frame.timestamp_s:6.1f}s | Machine state: {frame.machine_state}")
        print(
            f"Measured -> BT={frame.bt_c:6.2f}°C, ET={frame.et_c:6.2f}°C, "
            f"RoR={frame.ror_c_per_min:6.2f}°C/min, "
            f"Gas={frame.gas_pct:5.1f}%, Pressure={frame.drum_pressure_pa:6.1f} Pa"
        )
        print(
            f"Filtered -> RoR={filtered_ror_c_per_min:6.2f}°C/min"
        )
        print(
            f"Estimated -> Tb={estimated_state.Tb:6.2f}, "
            f"RoR={estimated_state.RoR * 60.0:6.2f}°C/min, "
            f"E_drum={estimated_state.E_drum:5.3f}, M={estimated_state.M:5.3f}, "
            f"P_int={estimated_state.P_int:5.3f}, "
            f"Mai={estimated_state.p_mai:5.3f}, Dev={estimated_state.p_dev:5.3f}"
        )

        if mpc_result.success:
            print(f"MPC objective: {mpc_result.objective_value:.4f} | status={mpc_result.status}")
        else:
            print(f"MPC fallback used | status={mpc_result.status}")

        active_alerts = alerts.active_labels()
        if active_alerts:
            print("Alerts:")
            print(f"  {', '.join(active_alerts)}")

        print("Recommendation:")
        print(f"  {recommendation.message}")

        print("Predicted flavor:")
        for k, v in best_eval.predicted_flavor.items():
            print(f"  {k}: {v:.3f}")

        print("Predicted structure:")
        for k, v in best_eval.structure_summary.items():
            print(f"  {k}: {v:.3f}")

        print(f"Flavor cost: {best_eval.flavor_cost:.4f}")

        # ------------------------------------------------------------
        # 9. Log step
        # ------------------------------------------------------------
        logger.log_step(
            frame=frame,
            estimated_state=estimated_state,
            current_control=current_control,
            recommendation=recommendation,
            mpc_success=mpc_result.success,
            mpc_objective=mpc_result.objective_value,
            mpc_status=mpc_result.status,
            alerts=alerts,
        )

        # ------------------------------------------------------------
        # 10. Simulate operator applying recommendation
        # ------------------------------------------------------------
        current_control = recommended_control
        gateway.apply_control(current_control)

    print(f"\nRuntime log saved to: {project_root / 'artifacts' / 'runtime_log.csv'}")


if __name__ == "__main__":
    run_dummy_live_loop(steps=20)