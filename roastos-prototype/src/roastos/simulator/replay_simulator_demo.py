from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.replay_validator import (
    replay_roast_dataframe,
    summarize_replay_metrics,
)
from roastos.simulator.sim_loader import load_simulator_params


def main():
    parser = argparse.ArgumentParser(
        description="Replay one roast from calibration_dataset.parquet through the RoastOS calibrated simulator."
    )
    parser.add_argument(
        "--model-json",
        required=True,
        help="Path to physics_model_v2_2.json",
    )
    parser.add_argument(
        "--timeseries-parquet",
        required=True,
        help="Path to calibration_dataset.parquet",
    )
    parser.add_argument(
        "--roast-id",
        default=None,
        help="Optional roast/session identifier to filter one roast. If omitted, the first roast is used.",
    )
    parser.add_argument(
        "--teacher-force-et",
        action="store_true",
        help="Use actual ET as simulator input during replay.",
    )
    parser.add_argument(
        "--teacher-force-ror",
        action="store_true",
        help="Use actual RoR as simulator input during replay.",
    )
    parser.add_argument(
        "--teacher-force-phase",
        action="store_true",
        help="Use actual phase labels from the calibration dataset during replay.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save detailed replay rows as CSV",
    )
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
    )
    metrics = summarize_replay_metrics(result)
    detail_df = pd.DataFrame(result.rows)

    if detail_df.empty:
        raise ValueError("Replay produced no rows.")

    roast_value = detail_df["roast_id"].iloc[0]
    print(f"Replay roast_id: {roast_value}")
    print(f"Replay rows: {len(detail_df) + 1}")
    print(f"Teacher-forced ET: {args.teacher_force_et}")
    print(f"Teacher-forced RoR: {args.teacher_force_ror}")
    print(f"Teacher-forced phase: {args.teacher_force_phase}")

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
        "actual_bt",
        "pred_bt",
        "actual_et",
        "pred_et",
        "actual_ror",
        "pred_ror",
        "pred_e_drum_raw",
        "pred_e_drum",
        "bt_error",
        "et_error",
        "ror_error",
    ]
    cols = [c for c in preferred_cols if c in detail_df.columns]
    print(detail_df[cols].tail(10).to_string(index=False))

    if args.save_csv:
        out_path = Path(args.save_csv)
        detail_df.to_csv(out_path, index=False)
        print(f"\nSaved replay details to: {out_path}")


if __name__ == "__main__":
    main()