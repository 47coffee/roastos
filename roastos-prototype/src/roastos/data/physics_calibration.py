from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


DATASET_PATH = Path("data/processed/calibration_dataset.parquet")
OUTPUT_PATH = Path("artifacts/models/physics_model.json")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise RuntimeError("Calibration dataset not found. Run dataset_builder first.")
    return pd.read_parquet(DATASET_PATH)


def prepare_training_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()

    df["bt_next"] = df.groupby("roast_id")["bt_c"].shift(-1)
    df["et_delta"] = df["et_c"] - df["bt_c"]

    print("\nMissing values before cleaning:")
    print(df[["gas", "pressure", "ror", "et_c", "bt_c", "bt_next"]].isna().sum())

    df = df.dropna(
        subset=[
            "bt_c",
            "bt_next",
            "gas",
            "pressure",
            "ror",
            "et_delta",
        ]
    ).copy()

    print(f"Training rows after cleaning: {len(df)}")

    # Target: one-step bean temperature increase
    y = (df["bt_next"] - df["bt_c"]).to_numpy(dtype=float)

    # Model:
    # dTb = intercept + alpha_gas*gas + beta_et*et_delta - gamma_pressure*pressure - delta_ror*ror
    #
    # To fit with nonnegative coefficients directly, move the minus signs into the design matrix:
    # dTb = intercept + alpha_gas*gas + beta_et*et_delta + gamma_pressure*(-pressure) + delta_ror*(-ror)
    X = np.column_stack([
        np.ones(len(df), dtype=float),                     # intercept
        df["gas"].to_numpy(dtype=float),                   # alpha_gas >= 0
        df["et_delta"].to_numpy(dtype=float),              # beta_et >= 0
        (-df["pressure"].to_numpy(dtype=float)),           # gamma_pressure >= 0
        (-df["ror"].to_numpy(dtype=float)),                # delta_ror >= 0
    ])

    return X, y


def fit_physics_model_bounded(X: np.ndarray, y: np.ndarray) -> dict:
    # Bounds:
    # intercept free, others nonnegative
    lower_bounds = np.array([-np.inf, 0.0, 0.0, 0.0, 0.0], dtype=float)
    upper_bounds = np.array([ np.inf, np.inf, np.inf, np.inf, np.inf], dtype=float)

    result = lsq_linear(X, y, bounds=(lower_bounds, upper_bounds), lsmr_tol="auto", verbose=1)

    if not result.success:
        raise RuntimeError(f"Bounded calibration failed: {result.message}")

    coef = result.x

    coeffs = {
        "intercept": float(coef[0]),
        "alpha_gas": float(coef[1]),
        "beta_et": float(coef[2]),
        "gamma_pressure": float(coef[3]),
        "delta_ror": float(coef[4]),
    }

    # Simple fit diagnostics
    y_hat = X @ coef
    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    print("\nBounded fit diagnostics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")

    return coeffs


def save_model(coeffs: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)

    print("\nPhysics model saved:")
    print(OUTPUT_PATH)


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print("Rows:", len(df))

    X, y = prepare_training_matrix(df)
    print("Training samples:", len(X))

    coeffs = fit_physics_model_bounded(X, y)

    print("\nEstimated bounded physics coefficients:\n")
    for k, v in coeffs.items():
        print(f"{k:15s} {v: .6f}")

    save_model(coeffs)


if __name__ == "__main__":
    main()