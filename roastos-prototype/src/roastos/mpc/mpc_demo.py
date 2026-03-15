from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from roastos.config import load_settings
from roastos.mpc.phase_aware_mpc import PhaseAwareMPC
from roastos.mpc.target_profile import TargetPoint, TargetTrajectory, TerminalTargets
from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.replay_validator import _normalize_replay_dataframe, _build_context
from roastos.simulator.sim_loader import load_simulator_params
from roastos.simulator.sim_types import RoastControl
from roastos.simulator.state_estimator import RoastStateEstimator


def _safe_float(v, default: float = 0.0) -> float:
    if pd.isna(v):
        return default
    return float(v)


def interpret_recommendation(
    actual_gas: float,
    rec_gas: float,
    actual_pressure: float,
    rec_pressure: float,
) -> str:
    """
    Convert numeric MPC output into roaster-friendly guidance.
    """
    gas_diff = float(rec_gas) - float(actual_gas)
    pressure_diff = float(rec_pressure) - float(actual_pressure)

    if abs(gas_diff) < 0.01:
        gas_text = "Hold gas"
    elif gas_diff > 0:
        gas_text = f"Increase gas slightly (+{gas_diff:.2f})"
    else:
        gas_text = f"Reduce gas slightly ({gas_diff:.2f})"

    if abs(pressure_diff) < 1.0:
        pressure_text = "Hold airflow"
    elif pressure_diff > 0:
        pressure_text = f"Increase airflow (+{pressure_diff:.1f} Pa)"
    else:
        pressure_text = f"Reduce airflow ({pressure_diff:.1f} Pa)"

    return gas_text + " | " + pressure_text


def _build_target_from_future_rows(
    df: pd.DataFrame,
    start_index: int,
    horizon_steps: int,
) -> TargetTrajectory:
    """
    Build a simple V4.0 target from future measured BT/ET values.

    This is intentionally pragmatic for smoke testing:
    MPC tries to track the future actual roast trajectory.
    """
    points: list[TargetPoint] = []

    last_row = df.iloc[min(start_index, len(df) - 1)]

    for k in range(1, horizon_steps + 1):
        idx = min(start_index + k, len(df) - 1)
        row = df.iloc[idx]
        points.append(
            TargetPoint(
                bt=_safe_float(row["bt_c"]),
                et=_safe_float(row["et_c"]),
                phase=str(row["phase"]),
            )
        )
        last_row = row

    terminal = TerminalTargets(
        drop_bt=_safe_float(last_row["bt_c"]),
        drop_weight_kg=None,
    )

    return TargetTrajectory(points=points, terminal=terminal)


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Run V4.0 MPC demo on one replay roast.")
    parser.add_argument("--model-json", default=str(settings.paths.model_artifact))
    parser.add_argument("--timeseries-parquet", default=str(settings.paths.calibration_dataset))
    parser.add_argument("--roast-id", default="PR-0173")
    parser.add_argument("--steps", type=int, default=60, help="How many online control steps to simulate.")
    parser.add_argument("--save-csv", default=str(settings.paths.replay_output_dir / "mpc_demo.csv"))
    args = parser.parse_args()

    model_path = Path(args.model_json)
    parquet_path = Path(args.timeseries_parquet)

    print(f"Loading model: {model_path}")
    params = load_simulator_params(model_path)
    simulator = CalibratedRoasterSimulator(params)
    estimator = RoastStateEstimator(simulator)
    mpc = PhaseAwareMPC(simulator)

    print(f"Loading calibration dataset parquet: {parquet_path}")
    raw_df = pd.read_parquet(parquet_path)
    replay_df = _normalize_replay_dataframe(raw_df, roast_id=args.roast_id)
    context = _build_context(replay_df)

    if len(replay_df) < 3:
        raise ValueError("Not enough rows for MPC demo.")

    first = replay_df.iloc[0]

    initial_control = RoastControl(
        gas=_safe_float(first["gas"]),
        pressure=_safe_float(first["pressure"]),
        drum_speed=_safe_float(first["drum_speed"], 0.65),
    )

    # Use zero latent raw for demo init; V3.2 observer will still correct state online.
    estimated_state = estimator.initialize(
        t_sec=_safe_float(first["time_s"]),
        measured_bt=_safe_float(first["bt_c"]),
        measured_et=_safe_float(first["et_c"]),
        measured_ror=_safe_float(first["ror"]),
        control=initial_control,
        e_drum_raw=0.0,
        context=context,
        phase=str(first["phase"]),
    )

    print("\nMPC demo initialization")
    print("-" * 40)
    print(f"Roast ID: {args.roast_id}")
    print(f"Start weight kg: {context.start_weight_kg}")
    print(f"Bean start temp C: {context.bean_start_temp_c}")
    print(f"Charge temp C: {context.charge_temp_c}")
    print(f"Initial estimated BT: {estimated_state.bt:.3f}")
    print(f"Initial estimated ET: {estimated_state.et:.3f}")
    print(f"Initial estimated RoR: {estimated_state.ror:.3f}")

    rows: list[dict] = []

    previous_control = initial_control
    max_steps = min(args.steps, len(replay_df) - 2)

    for i in range(max_steps):
        row_now = replay_df.iloc[i]
        row_next = replay_df.iloc[i + 1]

        measured_bt_next = _safe_float(row_next["bt_c"])
        measured_et_next = _safe_float(row_next["et_c"])
        measured_ror_next = _safe_float(row_next["ror"])

        phase_override = str(row_now["phase"])

        target = _build_target_from_future_rows(
            replay_df,
            start_index=i,
            horizon_steps=settings.mpc.horizon_steps,
        )

        online_result = mpc.observe_and_recommend(
            estimator=estimator,
            previous_state=estimated_state,
            previous_control=previous_control,
            measured_bt=measured_bt_next,
            measured_et=measured_et_next,
            measured_ror=measured_ror_next,
            target=target,
            context=context,
            phase_override=phase_override,
        )

        estimated_state = online_result.estimated_state
        recommendation = online_result.recommendation

        actual_gas_now = _safe_float(row_now["gas"])
        actual_pressure_now = _safe_float(row_now["pressure"])
        actual_drum_now = _safe_float(row_now["drum_speed"], 0.65)

        recommendation_text = interpret_recommendation(
            actual_gas_now,
            recommendation.control.gas,
            actual_pressure_now,
            recommendation.control.pressure,
        )

        rows.append(
            {
                "i": i,
                "time_s": _safe_float(row_next["time_s"]),
                "phase_actual": str(row_next["phase"]),
                "measured_bt": measured_bt_next,
                "measured_et": measured_et_next,
                "measured_ror": measured_ror_next,
                "estimated_bt": estimated_state.bt,
                "estimated_et": estimated_state.et,
                "estimated_ror": estimated_state.ror,
                "estimated_e_drum": estimated_state.e_drum,
                "estimated_m_burden": estimated_state.m_burden,
                "estimated_p_dry": estimated_state.p_dry,
                "estimated_p_mai": estimated_state.p_mai,
                "estimated_p_dev": estimated_state.p_dev,
                "actual_gas_now": actual_gas_now,
                "actual_pressure_now": actual_pressure_now,
                "actual_drum_now": actual_drum_now,
                "recommended_gas": recommendation.control.gas,
                "recommended_pressure": recommendation.control.pressure,
                "recommended_drum": recommendation.control.drum_speed,
                "recommendation_text": recommendation_text,
                "mpc_objective": recommendation.objective,
                "candidate_index": recommendation.best_index,
            }
        )

        previous_control = recommendation.control

    out_df = pd.DataFrame(rows)

    print("\nLast 10 MPC rows")
    print("-" * 40)
    if not out_df.empty:
        preview_cols = [
            "time_s",
            "phase_actual",
            "measured_bt",
            "estimated_bt",
            "measured_et",
            "estimated_et",
            "actual_gas_now",
            "recommended_gas",
            "actual_pressure_now",
            "recommended_pressure",
            "recommendation_text",
            "mpc_objective",
        ]
        preview_cols = [c for c in preview_cols if c in out_df.columns]
        print(out_df[preview_cols].tail(10).to_string(index=False))

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    out_df.to_csv(save_path, index=False)
    print(f"\nSaved MPC demo rows to: {save_path}")

    # Save diagnostic plot
    if not out_df.empty:
        plot_path = save_path.with_suffix(".png")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Bean temperature
        axes[0].plot(out_df["time_s"], out_df["measured_bt"], label="Measured BT")
        axes[0].plot(out_df["time_s"], out_df["estimated_bt"], label="Estimated BT")
        axes[0].set_ylabel("BT (°C)")
        axes[0].set_title("Measured vs Estimated Bean Temperature")
        axes[0].legend()
        axes[0].grid(True)

        # Environment temperature
        axes[1].plot(out_df["time_s"], out_df["measured_et"], label="Measured ET")
        axes[1].plot(out_df["time_s"], out_df["estimated_et"], label="Estimated ET")
        axes[1].set_ylabel("ET (°C)")
        axes[1].set_title("Measured vs Estimated Environment Temperature")
        axes[1].legend()
        axes[1].grid(True)

        # Controls and recommendations
        axes[2].plot(out_df["time_s"], out_df["actual_gas_now"], label="Actual Gas")
        axes[2].plot(out_df["time_s"], out_df["recommended_gas"], label="Recommended Gas")
        axes[2].plot(out_df["time_s"], out_df["actual_pressure_now"], label="Actual Pressure")
        axes[2].plot(out_df["time_s"], out_df["recommended_pressure"], label="Recommended Pressure")
        axes[2].set_ylabel("Control")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Actual vs MPC Recommended Controls")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        print(f"Saved MPC plot to: {plot_path}")


if __name__ == "__main__":
    main()

