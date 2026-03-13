from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


DEFAULT_DATASET_PATH = "data/processed/calibration_dataset.parquet"
DEFAULT_OUTPUT_PATH = "artifacts/models/physics_model_v1.json"


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


def prepare_training_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

    df = df.dropna(subset=required_cols).copy()

    print(f"Training rows after cleaning: {len(df)}")

    # Target: one-step bean temperature increase
    y = df["bt_delta"].to_numpy(dtype=float)

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

    # Keep sign assumptions explicit:
    # + gas, + lagged gas, + et_delta, + lagged et_delta
    # - pressure, - lagged pressure, - current ror
    X = np.column_stack([
        np.ones(len(df), dtype=float),
        df["gas"].to_numpy(dtype=float),
        df["gas_lag1"].to_numpy(dtype=float),
        df["et_delta"].to_numpy(dtype=float),
        df["et_delta_lag1"].to_numpy(dtype=float),
        (-df["pressure"].to_numpy(dtype=float)),
        (-df["pressure_lag1"].to_numpy(dtype=float)),
        (-df["ror"].to_numpy(dtype=float)),
    ])

    return X, y, feature_names


def fit_physics_model_bounded(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    # intercept free, all physical effect magnitudes nonnegative
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
        "model_version": "v1",
        "target": "bt_delta",
        "feature_names": feature_names,
    }

    for name, value in zip(feature_names, coef):
        coeffs[name] = float(value)

    y_hat = X @ coef
    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))

    coeffs["rmse"] = rmse
    coeffs["mae"] = mae
    coeffs["r2"] = r2
    coeffs["n_samples"] = int(len(y))

    print("\nBounded fit diagnostics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R^2 : {r2:.6f}")

    return coeffs


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

    X, y, feature_names = prepare_training_matrix(df)
    print("Training samples:", len(X))

    coeffs = fit_physics_model_bounded(X, y, feature_names)

    print("\nEstimated bounded physics coefficients (v1):\n")
    for name in feature_names:
        print(f"{name:18s} {coeffs[name]: .6f}")

    save_model(coeffs)


if __name__ == "__main__":
    main()
