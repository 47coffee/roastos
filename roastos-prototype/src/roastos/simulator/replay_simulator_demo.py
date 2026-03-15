from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roastos.config import load_settings
from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.replay_validator import (
    replay_roast_dataframe,
    summarize_replay_metrics,
)
from roastos.simulator.sim_loader import load_simulator_params


def _safe_first(detail_df: pd.DataFrame, col: str):
    if col in detail_df.columns and not detail_df.empty:
        val = detail_df[col].iloc[0]
        if pd.notna(val):
            return val
    return None


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Replay one roast from calibration_dataset.parquet through the RoastOS calibrated simulator."
    )
    parser.add_argument("--model-json", default=str(settings.paths.model_artifact))
    parser.add_argument("--timeseries-parquet", default=str(settings.paths.calibration_dataset))
    parser.add_argument("--roast-id", default=None)
    parser.add_argument("--teacher-force-et", action="store_true", default=settings.replay.teacher_force_et)
    parser.add_argument("--teacher-force-ror", action="store_true", default=settings.replay.teacher_force_ror)
    parser.add_argument("--teacher-force-phase", action="store_true", default=settings.replay.teacher_force_phase)
    parser.add_argument("--warmup-rows", type=int, default=settings.replay.warmup_rows)
    parser.add_argument("--use-estimator", action="store_true", help="Enable V3.2 state estimator.")
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()

    model_path = Path(args.model_json)
    parquet_path = Path(args.timeseries_parquet)

    print(f"Loading model: {model_path}")
    params = load_simulator_params(model_path)
    simulator = CalibratedRoasterSimulator(params)

    print(f"Loading calibration dataset parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    result = replay_roast_dataframe(
        df=df,
        simulator=simulator,
        roast_id=args.roast_id,
        teacher_force_et=args.teacher_force_et,
        teacher_force_ror=args.teacher_force_ror,
        teacher_force_phase=args.teacher_force_phase,
        warmup_rows=args.warmup_rows,
        use_estimator=args.use_estimator,
    )
    metrics = summarize_replay_metrics(result)
    detail_df = pd.DataFrame(result.rows)

    if detail_df.empty:
        raise ValueError("Replay produced no rows.")

    roast_value = detail_df["roast_id"].iloc[0]
    bean_start_temp = _safe_first(detail_df, "bean_start_temp_c")
    charge_temp = _safe_first(detail_df, "charge_temp_c")
    start_weight = _safe_first(detail_df, "start_weight_kg")

    print(f"Replay roast_id: {roast_value}")
    print(f"Replay rows: {len(detail_df) + 1}")
    print(f"Teacher-forced ET: {args.teacher_force_et}")
    print(f"Teacher-forced RoR: {args.teacher_force_ror}")
    print(f"Teacher-forced phase: {args.teacher_force_phase}")
    print(f"Observer enabled: {args.use_estimator}")
    print(f"Warmup rows: {args.warmup_rows}")
    print(f"Start weight kg: {start_weight}")
    print(f"Bean start temp C: {bean_start_temp}")
    print(f"Charge temp C: {charge_temp}")

    print("\nReplay metrics")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.6f}")
        else:
            print(f"{k:20s}: {v}")

    print("\nLast 10 replay rows")
    print("-" * 40)

    preferred_cols = [
        "time_s",
        "phase_used",
        "phase_actual_next",
        "gas",
        "pressure",
        "drum_speed",
        "observer_enabled",
        "start_weight_kg",
        "bean_start_temp_c",
        "charge_temp_c",
        "actual_bt",
        "pred_bt",
        "actual_et",
        "pred_et",
        "actual_ror",
        "pred_ror",
        "pred_e_drum_raw",
        "pred_e_drum",
        "pred_m_burden",
        "pred_p_dry",
        "pred_p_mai",
        "pred_p_dev",
        "bt_error",
        "et_error",
        "ror_error",
    ]
    cols = [c for c in preferred_cols if c in detail_df.columns]
    print(detail_df[cols].tail(10).to_string(index=False))

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(out_path, index=False)
        print(f"\nSaved replay details to: {out_path}")


if __name__ == "__main__":
    main()

