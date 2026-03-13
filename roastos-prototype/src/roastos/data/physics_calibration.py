from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


DEFAULT_DATASET_PATH = "data/processed/calibration_dataset.parquet"
DEFAULT_OUTPUT_PATH = "artifacts/models/physics_model_v2_2.json"
PHASES = ["drying", "maillard", "development"]


def _project_root() -> Path:
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


def ensure_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    if "bt_next" not in df.columns:
        df["bt_next"] = df.groupby("roast_id")["bt_c"].shift(-1)
    if "bt_delta" not in df.columns:
        df["bt_delta"] = df["bt_next"] - df["bt_c"]
    if "et_delta" not in df.columns:
        df["et_delta"] = df["et_c"] - df["bt_c"]
    if "et_delta_lag1" not in df.columns:
        df["et_delta_lag1"] = df.groupby("roast_id")["et_delta"].shift(1)
    if "gas_lag1" not in df.columns:
        df["gas_lag1"] = df.groupby("roast_id")["gas"].shift(1)
    if "pressure_lag1" not in df.columns:
        df["pressure_lag1"] = df.groupby("roast_id")["pressure"].shift(1)
    if "gas_delta" not in df.columns:
        df["gas_delta"] = df["gas"] - df["gas_lag1"]
    if "pressure_delta" not in df.columns:
        df["pressure_delta"] = df["pressure"] - df["pressure_lag1"]
    if "bt_c_norm" not in df.columns:
        df["bt_c_norm"] = df["bt_c"] / 200.0

    return df


def add_latent_drum_energy(
    df: pd.DataFrame,
    decay: float,
    pressure_scale: float,
) -> pd.DataFrame:
    df = ensure_v2_features(df)
    out = df.copy()

    if pressure_scale <= 0:
        raise ValueError("pressure_scale must be > 0")

    out["pressure_norm"] = out["pressure"] / pressure_scale
    out["e_drum_raw"] = 0.0

    for roast_id, idx in out.groupby("roast_id").groups.items():
        e_prev = 0.0
        ordered_idx = list(idx)
        for i in ordered_idx:
            gas = out.at[i, "gas"]
            pressure_norm = out.at[i, "pressure_norm"]

            gas = 0.0 if pd.isna(gas) else float(gas)
            pressure_norm = 0.0 if pd.isna(pressure_norm) else float(pressure_norm)

            e_now = decay * e_prev + gas - pressure_norm
            out.at[i, "e_drum_raw"] = e_now
            e_prev = e_now

    mean_val = out["e_drum_raw"].mean()
    std_val = out["e_drum_raw"].std()
    if pd.isna(std_val) or std_val <= 1e-9:
        out["e_drum"] = out["e_drum_raw"]
    else:
        out["e_drum"] = (out["e_drum_raw"] - mean_val) / std_val

    return out



def prepare_training_matrix_v2_2(
    df: pd.DataFrame,
    include_gas: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    df = df.copy()

    required_cols = [
        "bt_delta",
        "e_drum",
        "et_delta",
        "bt_c_norm",
        "ror",
        "pressure",
    ]
    if include_gas:
        required_cols.append("gas")

    print("\nMissing values before cleaning:")
    print(df[required_cols].isna().sum())

    df_clean = df.dropna(subset=required_cols).copy()
    print(f"Training rows after cleaning: {len(df_clean)}")

    y = df_clean["bt_delta"].to_numpy(dtype=float)

    feature_names = [
        "intercept",
        "e_drum",
        "et_delta",
        "neg_bt_level",
        "neg_ror",
        "neg_pressure_direct",
    ]

    columns = [
        np.ones(len(df_clean), dtype=float),
        df_clean["e_drum"].to_numpy(dtype=float),
        df_clean["et_delta"].to_numpy(dtype=float),
        (-df_clean["bt_c_norm"].to_numpy(dtype=float)),
        (-df_clean["ror"].to_numpy(dtype=float)),
        (-df_clean["pressure"].to_numpy(dtype=float)),
    ]

    if include_gas:
        feature_names.insert(2, "gas")
        columns.insert(2, df_clean["gas"].to_numpy(dtype=float))

    X = np.column_stack(columns)
    return X, y, feature_names, df_clean



def fit_bounded_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_version: str,
) -> dict:
    lower_bounds = np.array([-np.inf] + [0.0] * (X.shape[1] - 1), dtype=float)
    upper_bounds = np.array([np.inf] + [np.inf] * (X.shape[1] - 1), dtype=float)

    result = lsq_linear(
        X,
        y,
        bounds=(lower_bounds, upper_bounds),
        lsmr_tol="auto",
        verbose=0,
    )

    if not result.success:
        raise RuntimeError(f"Bounded calibration failed: {result.message}")

    coef = result.x
    y_hat = X @ coef
    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - np.sum(residuals ** 2) / tss) if tss > 0 else 0.0

    coeffs = {
        "model_version": model_version,
        "target": "bt_delta",
        "feature_names": feature_names,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_samples": int(len(y)),
    }
    for name, value in zip(feature_names, coef):
        coeffs[name] = float(value)

    return coeffs



def fit_phase_models_v2_2(
    df: pd.DataFrame,
    decay: float,
    pressure_scale: float,
    include_gas: bool,
) -> dict:
    phase_models: dict[str, dict] = {}

    for phase in PHASES:
        phase_df = df[df["phase"] == phase].copy()

        if phase_df.empty:
            phase_models[phase] = {
                "model_version": "v2.2",
                "phase": phase,
                "status": "skipped_empty_phase",
                "n_samples": 0,
            }
            continue

        X, y, feature_names, cleaned_df = prepare_training_matrix_v2_2(
            phase_df,
            include_gas=include_gas,
        )
        if len(X) < len(feature_names):
            phase_models[phase] = {
                "model_version": "v2.2",
                "phase": phase,
                "status": "skipped_insufficient_samples",
                "n_samples": int(len(X)),
            }
            continue

        coeffs = fit_bounded_regression(X, y, feature_names, model_version="v2.2")
        coeffs["phase"] = phase
        coeffs["status"] = "ok"
        coeffs["n_roasts"] = int(cleaned_df["roast_id"].nunique())
        coeffs["latent_decay"] = float(decay)
        coeffs["pressure_scale"] = float(pressure_scale)
        coeffs["include_gas"] = bool(include_gas)
        phase_models[phase] = coeffs

    return phase_models



def summarize_phase_models(phase_models: dict[str, dict]) -> dict:
    valid = [m for m in phase_models.values() if m.get("status") == "ok"]
    if not valid:
        return {
            "mean_r2": -np.inf,
            "weighted_r2": -np.inf,
            "total_samples": 0,
        }

    total_samples = sum(m["n_samples"] for m in valid)
    weighted_r2 = sum(m["r2"] * m["n_samples"] for m in valid) / total_samples
    mean_r2 = sum(m["r2"] for m in valid) / len(valid)
    return {
        "mean_r2": float(mean_r2),
        "weighted_r2": float(weighted_r2),
        "total_samples": int(total_samples),
    }



def search_model_config(df: pd.DataFrame) -> tuple[dict, dict]:
    pressure_median = float(df["pressure"].median()) if "pressure" in df.columns else 100.0
    pressure_scales = [max(10.0, pressure_median * mult) for mult in (0.5, 1.0, 1.5)]
    decay_grid = [0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99]
    gas_options = [False, True]

    best_score = -np.inf
    best_models: dict[str, dict] = {}
    best_config = {
        "decay": decay_grid[0],
        "pressure_scale": pressure_scales[0],
        "include_gas": False,
    }
    search_log = {}

    for include_gas in gas_options:
        for decay in decay_grid:
            for pressure_scale in pressure_scales:
                candidate_df = add_latent_drum_energy(
                    df,
                    decay=decay,
                    pressure_scale=pressure_scale,
                )
                candidate_models = fit_phase_models_v2_2(
                    candidate_df,
                    decay=decay,
                    pressure_scale=pressure_scale,
                    include_gas=include_gas,
                )
                summary = summarize_phase_models(candidate_models)
                score = summary["weighted_r2"]
                key = (
                    f"include_gas={include_gas}|"
                    f"decay={decay:.3f}|"
                    f"pressure_scale={pressure_scale:.3f}"
                )
                search_log[key] = summary

                if score > best_score:
                    best_score = score
                    best_models = candidate_models
                    best_config = {
                        "decay": float(decay),
                        "pressure_scale": float(pressure_scale),
                        "include_gas": bool(include_gas),
                    }

    best_summary = summarize_phase_models(best_models)
    return {
        "best_config": best_config,
        "best_models": best_models,
        "best_summary": best_summary,
    }, search_log



def save_model(coeffs: dict, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
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
    df = ensure_v2_features(df)
    print("Rows:", len(df))

    if "phase" in df.columns:
        print("Phase counts:")
        print(df["phase"].value_counts(dropna=False))
    else:
        raise KeyError("Dataset is missing required column: phase")

    print("\nSearching V2.2 model configurations...")
    search_result, search_log = search_model_config(df)
    best_config = search_result["best_config"]
    best_models = search_result["best_models"]
    best_summary = search_result["best_summary"]

    print(f"Best latent decay: {best_config['decay']:.3f}")
    print(f"Best pressure scale: {best_config['pressure_scale']:.3f}")
    print(f"Best include_gas: {best_config['include_gas']}")
    print("Search summary:", best_summary)

    for phase in PHASES:
        coeffs = best_models.get(phase, {})
        print("\n" + "=" * 70)
        print(f"PHASE: {phase.upper()}")
        print("=" * 70)
        if coeffs.get("status") != "ok":
            print(coeffs)
            continue

        print("Bounded fit diagnostics:")
        print(f"  RMSE: {coeffs['rmse']:.6f}")
        print(f"  MAE : {coeffs['mae']:.6f}")
        print(f"  R^2 : {coeffs['r2']:.6f}")
        print("\nEstimated bounded physics coefficients (v2.2):\n")
        for name in coeffs["feature_names"]:
            print(f"{name:18s} {coeffs[name]: .6f}")

    model_payload = {
        "model_version": "v2.2",
        "target": "bt_delta",
        "latent_state": {
            "name": "e_drum",
            "equation": "e_drum_t = decay * e_drum_t_minus_1 + gas - pressure / pressure_scale",
            "best_decay": float(best_config["decay"]),
            "best_pressure_scale": float(best_config["pressure_scale"]),
        },
        "best_config": best_config,
        "summary": best_summary,
        "phases": best_models,
        "hyperparameter_search": search_log,
    }

    save_model(model_payload)


if __name__ == "__main__":
    main()


