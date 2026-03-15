from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .settings import (
    CalibrationConfig,
    ContextModelConfig,
    EKFConfig,
    FlavourConfig,
    MPCConfig,
    PathsConfig,
    PhaseThresholdsConfig,
    ReplayConfig,
    RoastOSSettings,
    SimulatorConfig,
    StateModelConfig,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _legacy_or_new(raw: Dict[str, Any], new_key: str, legacy_key: str, default):
    if new_key in raw:
        return raw.get(new_key)
    if legacy_key in raw:
        return raw.get(legacy_key)
    return default


def load_settings(config_path: str | Path | None = None) -> RoastOSSettings:
    root = _project_root()
    default_path = root / "src" / "roastos" / "config" / "defaults.toml"

    payload = _load_toml(default_path)

    env_path = os.getenv("ROASTOS_CONFIG")
    chosen_path = Path(config_path) if config_path is not None else (Path(env_path) if env_path else None)

    if chosen_path is not None:
        chosen_path = chosen_path if chosen_path.is_absolute() else (root / chosen_path).resolve()
        if chosen_path.exists():
            override_payload = _load_toml(chosen_path)
            payload = _deep_merge(payload, override_payload)

    paths_raw = payload.get("paths", {})
    simulator_raw = payload.get("simulator", {})
    phase_raw = payload.get("phase_thresholds", {})
    replay_raw = payload.get("replay", {})
    calibration_raw = payload.get("calibration", {})
    state_model_raw = payload.get("state_model", {})
    context_model_raw = payload.get("context_model", {})
    ekf_raw = payload.get("ekf", {})
    mpc_raw = payload.get("mpc", {})
    flavour_raw = payload.get("flavour", {})

    paths = PathsConfig(
        project_root=root,
        processed_folder=_resolve_path(root, paths_raw["processed_folder"]),
        calibration_dataset=_resolve_path(root, paths_raw["calibration_dataset"]),
        model_artifact=_resolve_path(root, paths_raw["model_artifact"]),
        replay_output_dir=_resolve_path(root, paths_raw["replay_output_dir"]),
    )

    reference_bean_start_temp_c = _legacy_or_new(
        context_model_raw,
        "reference_bean_start_temp_c",
        "reference_start_temp_c",
        25.0,
    )
    default_bean_start_temp_c = _legacy_or_new(
        context_model_raw,
        "default_bean_start_temp_c",
        "default_start_temp_c",
        25.0,
    )

    settings = RoastOSSettings(
        paths=paths,
        simulator=SimulatorConfig(**simulator_raw),
        phase_thresholds=PhaseThresholdsConfig(**phase_raw),
        replay=ReplayConfig(
            teacher_force_et=bool(replay_raw.get("teacher_force_et", False)),
            teacher_force_ror=bool(replay_raw.get("teacher_force_ror", False)),
            teacher_force_phase=bool(replay_raw.get("teacher_force_phase", True)),
            default_roast_ids=tuple(replay_raw.get("default_roast_ids", [])),
            warmup_rows=int(replay_raw.get("warmup_rows", 120)),
        ),
        calibration=CalibrationConfig(
            bt_model_version=calibration_raw.get("bt_model_version", "v3.0"),
            et_model_version=calibration_raw.get("et_model_version", "et_v3.0"),
            release_label=calibration_raw.get("release_label", "V3.1"),
            release_notes=calibration_raw.get("release_notes", ""),
            phase_names=tuple(calibration_raw.get("phase_names", ["drying", "maillard", "development"])),
            latent_decay_grid=tuple(calibration_raw.get("latent_decay_grid", [0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99])),
            pressure_scale_multipliers=tuple(calibration_raw.get("pressure_scale_multipliers", [0.5, 1.0, 1.5])),
            include_gas_options=tuple(calibration_raw.get("include_gas_options", [False, True])),
        ),
        state_model=StateModelConfig(
            canonical_state_names=tuple(
                state_model_raw.get(
                    "canonical_state_names",
                    ["bt", "et", "ror", "e_drum", "m_burden", "p_dry", "p_mai", "p_dev"],
                )
            ),
            moisture_decay_rate=float(state_model_raw.get("moisture_decay_rate", 0.015)),
            moisture_heat_coeff=float(state_model_raw.get("moisture_heat_coeff", 0.002)),
            moisture_bt_drag_coeff=float(state_model_raw.get("moisture_bt_drag_coeff", 0.0)),
            moisture_et_drag_coeff=float(state_model_raw.get("moisture_et_drag_coeff", 0.0)),
            progress_drying_bt_start=float(state_model_raw.get("progress_drying_bt_start", 100.0)),
            progress_drying_bt_end=float(state_model_raw.get("progress_drying_bt_end", 160.0)),
            progress_maillard_bt_start=float(state_model_raw.get("progress_maillard_bt_start", 150.0)),
            progress_maillard_bt_end=float(state_model_raw.get("progress_maillard_bt_end", 196.0)),
            progress_development_bt_start=float(state_model_raw.get("progress_development_bt_start", 196.0)),
            progress_development_bt_end=float(state_model_raw.get("progress_development_bt_end", 222.0)),
        ),
        context_model=ContextModelConfig(
            enable_context_dynamics=bool(context_model_raw.get("enable_context_dynamics", False)),
            reference_charge_weight_kg=float(context_model_raw.get("reference_charge_weight_kg", 6.0)),
            reference_bean_start_temp_c=float(reference_bean_start_temp_c),
            default_bean_start_temp_c=(
                None if default_bean_start_temp_c is None else float(default_bean_start_temp_c)
            ),
            reference_charge_temp_c=float(context_model_raw.get("reference_charge_temp_c", 230.0)),
            default_charge_temp_c=(
                None
                if context_model_raw.get("default_charge_temp_c") is None
                else float(context_model_raw.get("default_charge_temp_c"))
            ),
            reference_start_temp_c=(
                None if reference_bean_start_temp_c is None else float(reference_bean_start_temp_c)
            ),
            default_start_temp_c=(
                None if default_bean_start_temp_c is None else float(default_bean_start_temp_c)
            ),
            max_context_response_scale=float(context_model_raw.get("max_context_response_scale", 1.35)),
            min_context_response_scale=float(context_model_raw.get("min_context_response_scale", 0.70)),
            default_start_weight_kg=(
                None
                if context_model_raw.get("default_start_weight_kg") is None
                else float(context_model_raw.get("default_start_weight_kg"))
            ),
            drop_weight_moisture_coeff=float(context_model_raw.get("drop_weight_moisture_coeff", 0.12)),
            drop_weight_development_coeff=float(context_model_raw.get("drop_weight_development_coeff", 0.05)),
            drop_weight_maillard_coeff=float(context_model_raw.get("drop_weight_maillard_coeff", 0.02)),
        ),
        ekf=EKFConfig(
            enabled=bool(ekf_raw.get("enabled", True)),
            k_bt=float(ekf_raw.get("k_bt", 0.35)),
            k_et=float(ekf_raw.get("k_et", 0.25)),
            k_ror=float(ekf_raw.get("k_ror", 0.20)),
        ),
        mpc=MPCConfig(
            horizon_steps=int(mpc_raw.get("horizon_steps", 20)),
            move_block=int(mpc_raw.get("move_block", 5)),
            gas_min=float(mpc_raw.get("gas_min", 0.0)),
            gas_max=float(mpc_raw.get("gas_max", 1.0)),
            pressure_min=float(mpc_raw.get("pressure_min", 0.0)),
            pressure_max=float(mpc_raw.get("pressure_max", 240.0)),
            gas_move_penalty=float(mpc_raw.get("gas_move_penalty", 2.0)),
            pressure_move_penalty=float(mpc_raw.get("pressure_move_penalty", 0.02)),
            bt_track_weight=float(mpc_raw.get("bt_track_weight", 1.0)),
            et_track_weight=float(mpc_raw.get("et_track_weight", 0.35)),
            terminal_bt_weight=float(mpc_raw.get("terminal_bt_weight", 1.25)),
            terminal_et_weight=float(mpc_raw.get("terminal_et_weight", 0.50)),
            terminal_drop_weight_weight=float(mpc_raw.get("terminal_drop_weight_weight", 1.0)),
        ),
        flavour=FlavourConfig(
            enabled=bool(flavour_raw.get("enabled", False)),
            predictor_mode=str(flavour_raw.get("predictor_mode", "stub")),
            default_weights=dict(
                flavour_raw.get(
                    "default_weights",
                    {
                        "sweetness": 1.0,
                        "clarity": 1.0,
                        "body": 1.0,
                        "acidity": 1.0,
                        "bitterness": 1.0,
                        "aroma": 1.0,
                    },
                )
            ),
        ),
        raw=payload,
    )

    return settings

