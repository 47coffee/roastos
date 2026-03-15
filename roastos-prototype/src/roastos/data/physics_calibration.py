from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from roastos.config import load_settings


_SETTINGS = load_settings()
PHASES = list(_SETTINGS.calibration.phase_names)

BT_MODEL_VERSION = _SETTINGS.calibration.bt_model_version
ET_MODEL_VERSION = _SETTINGS.calibration.et_model_version
RELEASE_LABEL = _SETTINGS.calibration.release_label
RELEASE_NOTES = _SETTINGS.calibration.release_notes


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path is not None else _SETTINGS.paths.calibration_dataset
    dataset_path = _resolve_project_path(path)

    if not dataset_path.exists():
        raise RuntimeError(
            f"Calibration dataset not found: {dataset_path}. Run dataset_builder first."
        )

    return pd.read_parquet(dataset_path)


def normalize_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "bean_start_temp_c" not in df.columns and "start_temp_c" in df.columns:
        df["bean_start_temp_c"] = df["start_temp_c"]

    return df


def ensure_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    if "bt_next" not in df.columns:
        df["bt_next"] = df.groupby("roast_id")["bt_c"].shift(-1)
    if "bt_delta" not in df.columns:
        df["bt_delta"] = df["bt_next"] - df["bt_c"]

    if "et_next" not in df.columns:
        df["et_next"] = df.groupby("roast_id")["et_c"].shift(-1)
    if "et_step" not in df.columns:
        df["et_step"] = df["et_next"] - df["et_c"]

    if "et_delta" not in df.columns:
        df["et_delta"] = df["et_c"] - df["bt_c"]
    if "bt_c_norm" not in df.columns:
        df["bt_c_norm"] = df["bt_c"] / 200.0
    if "et_c_norm" not in df.columns:
        df["et_c_norm"] = df["et_c"] / 250.0

    if "et_delta_lag1" not in df.columns:
        df["et_delta_lag1"] = df.groupby("roast_id")["et_delta"].shift(1)
    if "et_c_lag1" not in df.columns:
        df["et_c_lag1"] = df.groupby("roast_id")["et_c"].shift(1)
    if "gas_lag1" not in df.columns:
        df["gas_lag1"] = df.groupby("roast_id")["gas"].shift(1)
    if "airflow_lag1" not in df.columns and "airflow" in df.columns:
        df["airflow_lag1"] = df.groupby("roast_id")["airflow"].shift(1)
    if "pressure_lag1" not in df.columns:
        df["pressure_lag1"] = df.groupby("roast_id")["pressure"].shift(1)

    if "gas_delta" not in df.columns:
        df["gas_delta"] = df["gas"] - df["gas_lag1"]
    if "airflow_delta" not in df.columns and "airflow" in df.columns:
        df["airflow_delta"] = df["airflow"] - df["airflow_lag1"]
    if "pressure_delta" not in df.columns:
        df["pressure_delta"] = df["pressure"] - df["pressure_lag1"]

    return df


def add_latent_drum_energy(df: pd.DataFrame, decay: float, pressure_scale: float) -> pd.DataFrame:
    df = ensure_v2_features(df)
    out = df.copy()

    if pressure_scale <= 0:
        raise ValueError("pressure_scale must be > 0")

    out["pressure_norm"] = out["pressure"] / pressure_scale
    out["e_drum_raw"] = 0.0

    for _, idx in out.groupby("roast_id").groups.items():
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


def compute_latent_stats(df: pd.DataFrame, decay: float, pressure_scale: float) -> dict:
    tmp = add_latent_drum_energy(df, decay=decay, pressure_scale=pressure_scale)
    mean_val = float(tmp["e_drum_raw"].mean())
    std_val = float(tmp["e_drum_raw"].std())
    if not np.isfinite(std_val) or std_val <= 1e-9:
        std_val = 1.0
    return {"raw_mean": mean_val, "raw_std": std_val}


def _clean_coeff_dict(coeffs: dict, eps: float = 1e-12) -> dict:
    out = {}
    for k, v in coeffs.items():
        if isinstance(v, float) and abs(v) < eps:
            out[k] = 0.0
        else:
            out[k] = v
    return out


def prepare_training_matrix_v3_0(
    df: pd.DataFrame,
    include_gas: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    df = df.copy()

    required_cols = ["bt_delta", "e_drum", "et_delta", "bt_c_norm", "ror", "pressure"]
    if include_gas:
        required_cols.append("gas")

    print("\nMissing BT-model values before cleaning:")
    print(df[required_cols].isna().sum())

    df_clean = df.dropna(subset=required_cols).copy()
    print(f"BT-model training rows after cleaning: {len(df_clean)}")

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


def prepare_training_matrix_et_v3(
    df: pd.DataFrame,
    include_gas: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    df = df.copy()

    required_cols = [
        "et_step", "e_drum", "et_delta", "et_delta_lag1",
        "pressure", "pressure_lag1", "pressure_delta", "ror", "et_c_norm",
    ]
    if include_gas:
        required_cols.append("gas")

    print("\nMissing ET-model values before cleaning:")
    print(df[required_cols].isna().sum())

    df_clean = df.dropna(subset=required_cols).copy()
    print(f"ET-model training rows after cleaning: {len(df_clean)}")

    y = df_clean["et_step"].to_numpy(dtype=float)

    feature_names = [
        "intercept",
        "e_drum",
        "neg_et_bt_gap",
        "neg_et_bt_gap_lag1",
        "neg_pressure",
        "neg_pressure_lag1",
        "pressure_delta_pos",
        "neg_ror",
        "neg_et_level",
    ]

    columns = [
        np.ones(len(df_clean), dtype=float),
        df_clean["e_drum"].to_numpy(dtype=float),
        (-df_clean["et_delta"].to_numpy(dtype=float)),
        (-df_clean["et_delta_lag1"].to_numpy(dtype=float)),
        (-df_clean["pressure"].to_numpy(dtype=float)),
        (-df_clean["pressure_lag1"].to_numpy(dtype=float)),
        np.maximum(df_clean["pressure_delta"].to_numpy(dtype=float), 0.0),
        (-df_clean["ror"].to_numpy(dtype=float)),
        (-df_clean["et_c_norm"].to_numpy(dtype=float)),
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
    target_name: str,
) -> dict:
    lower_bounds = np.array([-np.inf] + [0.0] * (X.shape[1] - 1), dtype=float)
    upper_bounds = np.array([np.inf] + [np.inf] * (X.shape[1] - 1), dtype=float)

    result = lsq_linear(X, y, bounds=(lower_bounds, upper_bounds), lsmr_tol="auto", verbose=0)

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
        "target": target_name,
        "feature_names": feature_names,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_samples": int(len(y)),
    }
    for name, value in zip(feature_names, coef):
        coeffs[name] = float(value)

    return _clean_coeff_dict(coeffs)


def fit_phase_models_v3_0(df: pd.DataFrame, decay: float, pressure_scale: float, include_gas: bool) -> dict:
    phase_models: dict[str, dict] = {}

    for phase in PHASES:
        phase_df = df[df["phase"] == phase].copy()

        if phase_df.empty:
            phase_models[phase] = {
                "model_version": BT_MODEL_VERSION,
                "phase": phase,
                "status": "skipped_empty_phase",
                "n_samples": 0,
            }
            continue

        X, y, feature_names, cleaned_df = prepare_training_matrix_v3_0(phase_df, include_gas=include_gas)
        if len(X) < len(feature_names):
            phase_models[phase] = {
                "model_version": BT_MODEL_VERSION,
                "phase": phase,
                "status": "skipped_insufficient_samples",
                "n_samples": int(len(X)),
            }
            continue

        coeffs = fit_bounded_regression(X, y, feature_names, model_version=BT_MODEL_VERSION, target_name="bt_delta")
        coeffs["phase"] = phase
        coeffs["status"] = "ok"
        coeffs["n_roasts"] = int(cleaned_df["roast_id"].nunique())
        coeffs["latent_decay"] = float(decay)
        coeffs["pressure_scale"] = float(pressure_scale)
        coeffs["include_gas"] = bool(include_gas)
        phase_models[phase] = coeffs

    return phase_models


def fit_phase_et_models_v3(df: pd.DataFrame, decay: float, pressure_scale: float, include_gas: bool) -> dict:
    phase_models: dict[str, dict] = {}

    for phase in PHASES:
        phase_df = df[df["phase"] == phase].copy()

        if phase_df.empty:
            phase_models[phase] = {
                "model_version": ET_MODEL_VERSION,
                "phase": phase,
                "status": "skipped_empty_phase",
                "n_samples": 0,
            }
            continue

        X, y, feature_names, cleaned_df = prepare_training_matrix_et_v3(phase_df, include_gas=include_gas)
        if len(X) < len(feature_names):
            phase_models[phase] = {
                "model_version": ET_MODEL_VERSION,
                "phase": phase,
                "status": "skipped_insufficient_samples",
                "n_samples": int(len(X)),
            }
            continue

        coeffs = fit_bounded_regression(X, y, feature_names, model_version=ET_MODEL_VERSION, target_name="et_step")
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
        return {"mean_r2": -np.inf, "weighted_r2": -np.inf, "total_samples": 0}

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
    pressure_scales = [
        max(10.0, pressure_median * mult)
        for mult in _SETTINGS.calibration.pressure_scale_multipliers
    ]
    decay_grid = list(_SETTINGS.calibration.latent_decay_grid)
    gas_options = list(_SETTINGS.calibration.include_gas_options)

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
                candidate_df = add_latent_drum_energy(df, decay=decay, pressure_scale=pressure_scale)
                candidate_models = fit_phase_models_v3_0(
                    candidate_df, decay=decay, pressure_scale=pressure_scale, include_gas=include_gas
                )
                summary = summarize_phase_models(candidate_models)
                score = summary["weighted_r2"]
                key = f"include_gas={include_gas}|decay={decay:.3f}|pressure_scale={pressure_scale:.3f}"
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


def save_model(coeffs: dict, output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path is not None else _SETTINGS.paths.model_artifact
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)

    print("\nPhysics model saved:")
    print(output_path)
    return output_path


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    df = normalize_context_columns(df)
    df = ensure_v2_features(df)
    print("Rows:", len(df))

    if "phase" in df.columns:
        print("Phase counts:")
        print(df["phase"].value_counts(dropna=False))
    else:
        raise KeyError("Dataset is missing required column: phase")

    context_cols = [
        c for c in [
            "start_weight_kg",
            "bean_start_temp_c",
            "charge_temp_c",
            "actual_drop_bt",
            "actual_drop_weight_kg",
            "airflow",
            "airflow_pct",
            "drum_speed",
            "drum_speed_pct",
        ]
        if c in df.columns
    ]
    print("\nAvailable context/control columns:")
    print(context_cols if context_cols else "None")

    print(f"\nSearching {RELEASE_LABEL} BT model configurations...")
    search_result, search_log = search_model_config(df)
    best_config = search_result["best_config"]
    best_models = search_result["best_models"]
    best_summary = search_result["best_summary"]

    latent_stats = compute_latent_stats(
        df, decay=best_config["decay"], pressure_scale=best_config["pressure_scale"]
    )

    best_latent_df = add_latent_drum_energy(
        df, decay=best_config["decay"], pressure_scale=best_config["pressure_scale"]
    )

    et_models = fit_phase_et_models_v3(
        best_latent_df,
        decay=best_config["decay"],
        pressure_scale=best_config["pressure_scale"],
        include_gas=best_config["include_gas"],
    )
    et_summary = summarize_phase_models(et_models)

    print(f"Best latent decay: {best_config['decay']:.3f}")
    print(f"Best pressure scale: {best_config['pressure_scale']:.3f}")
    print(f"Best include_gas: {best_config['include_gas']}")
    print(f"{RELEASE_LABEL} BT search summary:", best_summary)

    for phase in PHASES:
        coeffs = best_models.get(phase, {})
        print("\n" + "=" * 70)
        print(f"BT PHASE: {phase.upper()}")
        print("=" * 70)
        if coeffs.get("status") != "ok":
            print(coeffs)
            continue

        print("Bounded fit diagnostics:")
        print(f"  RMSE: {coeffs['rmse']:.6f}")
        print(f"  MAE : {coeffs['mae']:.6f}")
        print(f"  R^2 : {coeffs['r2']:.6f}")
        print(f"\nEstimated bounded BT coefficients ({BT_MODEL_VERSION}):\n")
        for name in coeffs["feature_names"]:
            print(f"{name:22s} {coeffs[name]: .6f}")

    print("\n" + "#" * 70)
    print(f"ET STEP MODELS ({ET_MODEL_VERSION})")
    print("#" * 70)
    print("ET summary:", et_summary)

    for phase in PHASES:
        coeffs = et_models.get(phase, {})
        print("\n" + "-" * 70)
        print(f"ET PHASE: {phase.upper()}")
        print("-" * 70)
        if coeffs.get("status") != "ok":
            print(coeffs)
            continue

        print("Bounded fit diagnostics:")
        print(f"  RMSE: {coeffs['rmse']:.6f}")
        print(f"  MAE : {coeffs['mae']:.6f}")
        print(f"  R^2 : {coeffs['r2']:.6f}")
        print(f"\nEstimated bounded ET coefficients ({ET_MODEL_VERSION}):\n")
        for name in coeffs["feature_names"]:
            print(f"{name:22s} {coeffs[name]: .6f}")

    model_payload = {
        "release": {
            "label": RELEASE_LABEL,
            "notes": RELEASE_NOTES,
            "intended_use": "Replay-stable simulator baseline for phase-aware MPC development",
        },
        "model_version": BT_MODEL_VERSION,
        "target": "bt_delta",
        "feature_engineering": {
            "gas_source": "Curve - Gas -> gas_pct / 100.0",
            "airflow_source": "Curve - Airflow -> airflow_pct / 100.0",
            "drum_speed_source": "Curve - Drum speed -> drum_speed_pct / 100.0",
            "pressure_source": "Curve - Drum pressure -> drum_pressure_pa",
            "bt_source": "Curve - Bean temperature -> bt_c",
            "et_source": "Curve - Exhaust temperature -> et_c",
            "bt_c_norm_equation": "bt_c / 200.0",
            "et_c_norm_equation": "et_c / 250.0",
            "phase_source": "dataset phase column from first_crack-relative timing",
            "ror_source": "groupby(roast_id) diff(bt_c) / diff(time_s) * 60",
            "bt_target": "bt_next - bt_c",
            "et_target": "et_next - et_c",
        },
        "context_support": {
            "available_columns": context_cols,
            "note": (
                "Frozen V3.2 keeps the historical V3.0 regression structure. "
                "Gas is modeled as a direct control feature where enabled; drum pressure remains "
                "the pressure-side feature used in the plant. Airflow and drum speed are carried in the "
                "dataset contract for V4.1+ control/context integration."
            ),
            "bean_start_temp_definition": "bean temperature at charge",
            "charge_temp_definition": "machine / ET-side charge temperature near t=0",
        },
        "latent_state": {
            "name": "e_drum",
            "equation_raw": "e_drum_raw_t = decay * e_drum_raw_t_minus_1 + gas - pressure / pressure_scale",
            "equation_standardized": "e_drum = (e_drum_raw - raw_mean) / raw_std",
            "best_decay": float(best_config["decay"]),
            "best_pressure_scale": float(best_config["pressure_scale"]),
            "raw_mean": float(latent_stats["raw_mean"]),
            "raw_std": float(latent_stats["raw_std"]),
        },
        "best_config": best_config,
        "summary": best_summary,
        "phases": best_models,
        "et_models_summary": et_summary,
        "et_models": et_models,
        "hyperparameter_search": search_log,
    }

    save_model(model_payload)


if __name__ == "__main__":
    main()
