from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


DEFAULT_DATASET_PATH = "data/processed/calibration_dataset.parquet"
DEFAULT_OUTPUT_PATH = "artifacts/models/physics_model_v1_1.json"
PHASES = ["drying", "maillard", "development"]


def _project_root() -> Path:
    # .../roastos-prototype/src/roastos/data/physics_calibration.py
    # -> parents[3] = .../roastos-prototype
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def load_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    dataset_path = _resolve_project_path(dataset_path)

    if not dataset_path.exists():
        raise RuntimeError(
            f"Calibration dataset not found: {dataset_path}. "
            f"Run dataset_builder first."
        )

    return pd.read_parquet(dataset_path)


def prepare_training_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    df = df.copy()

    # Defensive fallbacks in case dataset_builder has not been re-run yet.
    if "bt_next" not in df.columns:
        df["bt_next"] = df.groupby("roast_id")["bt_c"].shift(-1)
    if "bt_delta" not in df.columns:
        df["bt_delta"] = df["bt_next"] - df["bt_c"]
    if "et_delta" not in df.columns:
        df["et_delta"] = df["et_c"] - df["bt_c"]
    if "gas_lag1" not in df.columns:
        df["gas_lag1"] = df.groupby("roast_id")["gas"].shift(1)
    if "pressure_lag1" not in df.columns:
        df["pressure_lag1"] = df.groupby("roast_id")["pressure"].shift(1)
    if "et_delta_lag1" not in df.columns:
        df["et_delta_lag1"] = df.groupby("roast_id")["et_delta"].shift(1)

    required_cols = [
        "bt_c",
        "bt_next",
        "bt_delta",
        "gas",
        "gas_lag1",
        "pressure",
        "pressure_lag1",
        "ror",
        "et_delta",
        "et_delta_lag1",
    ]

    print("\nMissing values before cleaning:")
    print(df[required_cols].isna().sum())

    df_clean = df.dropna(subset=required_cols).copy()

    print(f"Training rows after cleaning: {len(df_clean)}")

    y = df_clean["bt_delta"].to_numpy(dtype=float)

    feature_names = [
        "intercept",
        "gas",
        "gas_lag1",
        "et_delta",
        "et_delta_lag1",
        "neg_pressure",
        "neg_pressure_lag1",
        "neg_ror",
    ]

    X = np.column_stack([
        np.ones(len(df_clean), dtype=float),
        df_clean["gas"].to_numpy(dtype=float),
        df_clean["gas_lag1"].to_numpy(dtype=float),
        df_clean["et_delta"].to_numpy(dtype=float),
        df_clean["et_delta_lag1"].to_numpy(dtype=float),
        (-df_clean["pressure"].to_numpy(dtype=float)),
        (-df_clean["pressure_lag1"].to_numpy(dtype=float)),
        (-df_clean["ror"].to_numpy(dtype=float)),
    ])

    return X, y, feature_names, df_clean


def fit_physics_model_bounded(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_version: str = "v1.1",
) -> dict:
    lower_bounds = np.array([-np.inf] + [0.0] * (X.shape[1] - 1), dtype=float)
    upper_bounds = np.array([np.inf] + [np.inf] * (X.shape[1] - 1), dtype=float)

    result = lsq_linear(
        X,
        y,
        bounds=(lower_bounds, upper_bounds),
        lsmr_tol="auto",
        verbose=1,
    )

    if not result.success:
        raise RuntimeError(f"Bounded calibration failed: {result.message}")

    coef = result.x

    coeffs = {
        "model_version": model_version,
        "target": "bt_delta",
        "feature_names": feature_names,
    }

    for name, value in zip(feature_names, coef):
        coeffs[name] = float(value)

    y_hat = X @ coef
    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - np.sum(residuals ** 2) / tss) if tss > 0 else 0.0

    coeffs["rmse"] = rmse
    coeffs["mae"] = mae
    coeffs["r2"] = r2
    coeffs["n_samples"] = int(len(y))

    print("\nBounded fit diagnostics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R^2 : {r2:.6f}")

    return coeffs


def fit_phase_models(df: pd.DataFrame) -> dict:
    if "phase" not in df.columns:
        raise KeyError("Dataset is missing required column: phase")

    models: dict[str, dict] = {}

    for phase in PHASES:
        print("\n" + "=" * 70)
        print(f"PHASE: {phase.upper()}")
        print("=" * 70)

        phase_df = df[df["phase"] == phase].copy()
        print(f"Rows in raw phase subset: {len(phase_df)}")

        if phase_df.empty:
            print(f"[WARN] No rows found for phase '{phase}'. Skipping.")
            models[phase] = {
                "model_version": "v1.1",
                "phase": phase,
                "status": "skipped_empty_phase",
                "n_samples": 0,
            }
            continue

        X, y, feature_names, cleaned_df = prepare_training_matrix(phase_df)
        print(f"Training samples for {phase}: {len(X)}")

        if len(X) < len(feature_names):
            print(f"[WARN] Not enough cleaned samples for phase '{phase}'. Skipping.")
            models[phase] = {
                "model_version": "v1.1",
                "phase": phase,
                "status": "skipped_insufficient_samples",
                "n_samples": int(len(X)),
            }
            continue

        coeffs = fit_physics_model_bounded(
            X,
            y,
            feature_names,
            model_version="v1.1",
        )
        coeffs["phase"] = phase
        coeffs["status"] = "ok"
        coeffs["n_roasts"] = int(cleaned_df["roast_id"].nunique())

        print(f"\nEstimated bounded physics coefficients ({phase}):\n")
        for name in feature_names:
            print(f"{name:18s} {coeffs[name]: .6f}")

        models[phase] = coeffs

    return models


def save_model(
    coeffs: dict,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path = _resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)

    print("\nPhysics model saved:")
    print(output_path)

    return output_path


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print("Rows:", len(df))

    if "phase" in df.columns:
        print("Phase counts:")
        print(df["phase"].value_counts(dropna=False))
    else:
        print("[WARN] No phase column found in dataset.")

    phase_models = fit_phase_models(df)
    save_model(phase_models)


if __name__ == "__main__":
    main()
