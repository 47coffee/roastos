from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roastos.config import load_settings
from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.replay_validator import replay_roast_dataframe, summarize_replay_metrics
from roastos.simulator.sim_loader import load_simulator_params


def _safe_first(detail_df: pd.DataFrame, col: str):
    if col in detail_df.columns and not detail_df.empty:
        val = detail_df[col].iloc[0]
        if pd.notna(val):
            return val
    return None


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Run replay benchmark across multiple roast IDs.")
    parser.add_argument("--model-json", default=str(settings.paths.model_artifact))
    parser.add_argument("--timeseries-parquet", default=str(settings.paths.calibration_dataset))
    parser.add_argument("--roast-ids", nargs="*", default=list(settings.replay.default_roast_ids))
    parser.add_argument("--teacher-force-et", action="store_true", default=settings.replay.teacher_force_et)
    parser.add_argument("--teacher-force-ror", action="store_true", default=settings.replay.teacher_force_ror)
    parser.add_argument("--teacher-force-phase", action="store_true", default=settings.replay.teacher_force_phase)
    parser.add_argument("--warmup-rows", type=int, default=settings.replay.warmup_rows)
    parser.add_argument("--use-estimator", action="store_true", help="Enable V3.2 state estimator.")
    parser.add_argument("--save-csv", default=str(settings.paths.replay_output_dir / "batch_replay_metrics.csv"))
    args = parser.parse_args()

    params = load_simulator_params(args.model_json)
    simulator = CalibratedRoasterSimulator(params)
    df = pd.read_parquet(args.timeseries_parquet)

    rows = []
    for roast_id in args.roast_ids:
        try:
            result = replay_roast_dataframe(
                df=df,
                simulator=simulator,
                roast_id=roast_id,
                teacher_force_et=args.teacher_force_et,
                teacher_force_ror=args.teacher_force_ror,
                teacher_force_phase=args.teacher_force_phase,
                warmup_rows=args.warmup_rows,
                use_estimator=args.use_estimator,
            )
            metrics = summarize_replay_metrics(result)
            detail_df = pd.DataFrame(result.rows)

            metrics["roast_id"] = roast_id
            metrics["observer_enabled"] = bool(args.use_estimator)
            metrics["start_weight_kg"] = _safe_first(detail_df, "start_weight_kg")
            metrics["bean_start_temp_c"] = _safe_first(detail_df, "bean_start_temp_c")
            metrics["charge_temp_c"] = _safe_first(detail_df, "charge_temp_c")
            metrics["model_json"] = str(args.model_json)

            rows.append(metrics)
            print(f"[OK] {roast_id}")
        except Exception as e:
            rows.append(
                {
                    "roast_id": roast_id,
                    "observer_enabled": bool(args.use_estimator),
                    "model_json": str(args.model_json),
                    "status": f"failed: {e}",
                }
            )
            print(f"[FAIL] {roast_id}: {e}")

    out_df = pd.DataFrame(rows)
    print("\nReplay benchmark summary:")
    print(out_df.to_string(index=False))

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_path, index=False)
    print(f"\nSaved benchmark metrics to: {save_path}")


if __name__ == "__main__":
    main()

