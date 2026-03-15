from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from roastos.config import load_settings
from .sim_types import (
    PhaseBTParams,
    PhaseETParams,
    PhaseModelParams,
    SimulatorParams,
    PHASES,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _f(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _clean_small(x: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(x) < eps else float(x)


def _legacy_or_new(raw: Dict[str, Any], new_key: str, legacy_key: str, default: float) -> float:
    if new_key in raw and raw.get(new_key) is not None:
        return float(raw.get(new_key))
    if legacy_key in raw and raw.get(legacy_key) is not None:
        return float(raw.get(legacy_key))
    return float(default)


def load_simulator_params(model_json_path: str | Path | None = None) -> SimulatorParams:
    settings = load_settings()
    path = Path(model_json_path) if model_json_path is not None else settings.paths.model_artifact
    payload = _read_json(path)

    phases_payload = payload.get("phases", {})
    et_models_payload = payload.get("et_models", {})
    latent_payload = payload.get("latent_state", {})
    best_config = payload.get("best_config", {})

    if not isinstance(phases_payload, dict):
        raise ValueError("Expected 'phases' in model JSON.")
    if not isinstance(et_models_payload, dict):
        et_models_payload = {}

    state_model_raw = settings.raw.get("state_model", {}) if isinstance(settings.raw, dict) else {}
    context_model_raw = settings.raw.get("context_model", {}) if isinstance(settings.raw, dict) else {}

    params = SimulatorParams()

    params.dt_sec = float(settings.simulator.dt_sec)
    params.bt_norm_denominator = float(settings.simulator.bt_norm_denominator)
    params.bt_norm_offset = float(settings.simulator.bt_norm_offset)
    params.et_norm_denominator = float(settings.simulator.et_norm_denominator)
    params.ror_filter_alpha = float(settings.simulator.ror_filter_alpha)
    params.ror_model_clip = float(settings.simulator.ror_model_clip)

    params.gas_already_normalized = bool(settings.simulator.gas_already_normalized)
    params.pressure_is_raw_pa = bool(settings.simulator.pressure_is_raw_pa)

    params.latent_decay = float(best_config.get("decay", latent_payload.get("best_decay", 0.99)))
    params.latent_pressure_scale = float(
        best_config.get("pressure_scale", latent_payload.get("best_pressure_scale", 100.0))
    )
    params.latent_mean = float(latent_payload.get("raw_mean", 0.0))
    params.latent_std = float(latent_payload.get("raw_std", 1.0))
    if abs(params.latent_std) < 1e-9:
        params.latent_std = 1.0

    params.include_gas_feature = bool(best_config.get("include_gas", False))

    # V4 state-model hooks
    params.moisture_decay_rate = float(state_model_raw.get("moisture_decay_rate", params.moisture_decay_rate))
    params.moisture_heat_coeff = float(state_model_raw.get("moisture_heat_coeff", params.moisture_heat_coeff))
    params.moisture_bt_drag_coeff = float(state_model_raw.get("moisture_bt_drag_coeff", params.moisture_bt_drag_coeff))
    params.moisture_et_drag_coeff = float(state_model_raw.get("moisture_et_drag_coeff", params.moisture_et_drag_coeff))

    params.progress_drying_bt_start = float(
        state_model_raw.get("progress_drying_bt_start", params.progress_drying_bt_start)
    )
    params.progress_drying_bt_end = float(
        state_model_raw.get("progress_drying_bt_end", params.progress_drying_bt_end)
    )
    params.progress_maillard_bt_start = float(
        state_model_raw.get("progress_maillard_bt_start", params.progress_maillard_bt_start)
    )
    params.progress_maillard_bt_end = float(
        state_model_raw.get("progress_maillard_bt_end", params.progress_maillard_bt_end)
    )
    params.progress_development_bt_start = float(
        state_model_raw.get("progress_development_bt_start", params.progress_development_bt_start)
    )
    params.progress_development_bt_end = float(
        state_model_raw.get("progress_development_bt_end", params.progress_development_bt_end)
    )

    # V4.1 context-aware hooks
    params.enable_context_dynamics = bool(
        context_model_raw.get("enable_context_dynamics", params.enable_context_dynamics)
    )
    params.reference_charge_weight_kg = float(
        context_model_raw.get("reference_charge_weight_kg", params.reference_charge_weight_kg)
    )

    params.reference_bean_start_temp_c = _legacy_or_new(
        context_model_raw,
        "reference_bean_start_temp_c",
        "reference_start_temp_c",
        params.reference_bean_start_temp_c,
    )
    params.reference_charge_temp_c = float(
        context_model_raw.get("reference_charge_temp_c", params.reference_charge_temp_c)
    )

    # backward compatibility alias only
    params.reference_start_temp_c = params.reference_bean_start_temp_c

    params.max_context_response_scale = float(
        context_model_raw.get("max_context_response_scale", params.max_context_response_scale)
    )
    params.min_context_response_scale = float(
        context_model_raw.get("min_context_response_scale", params.min_context_response_scale)
    )
    params.drop_weight_moisture_coeff = float(
        context_model_raw.get("drop_weight_moisture_coeff", params.drop_weight_moisture_coeff)
    )
    params.drop_weight_development_coeff = float(
        context_model_raw.get("drop_weight_development_coeff", params.drop_weight_development_coeff)
    )
    params.drop_weight_maillard_coeff = float(
        context_model_raw.get("drop_weight_maillard_coeff", params.drop_weight_maillard_coeff)
    )

    phase_models: Dict[str, PhaseModelParams] = {}

    for phase in PHASES:
        bt_ph = phases_payload.get(phase, {})
        if not isinstance(bt_ph, dict):
            bt_ph = {}

        et_ph = et_models_payload.get(phase, {})
        if not isinstance(et_ph, dict):
            et_ph = {}

        bt_feature_names = bt_ph.get("feature_names", [])
        et_feature_names = et_ph.get("feature_names", [])

        model = PhaseModelParams(
            bt=PhaseBTParams(
                intercept=_clean_small(_f(bt_ph, "intercept", 0.0)),
                c_e_drum=_clean_small(_f(bt_ph, "e_drum", 0.0)),
                c_et_delta=_clean_small(_f(bt_ph, "et_delta", 0.0)),
                c_bt_level=_clean_small(-_f(bt_ph, "neg_bt_level", 0.0)),
                c_ror=_clean_small(-_f(bt_ph, "neg_ror", 0.0)),
                c_pressure_direct=_clean_small(-_f(bt_ph, "neg_pressure_direct", 0.0)),
                c_gas=_clean_small(_f(bt_ph, "gas", 0.0)) if "gas" in bt_feature_names else 0.0,
            ),
            et=PhaseETParams(
                intercept=_clean_small(_f(et_ph, "intercept", 0.0)),
                c_e_drum=_clean_small(_f(et_ph, "e_drum", 0.0)),
                c_gas=_clean_small(_f(et_ph, "gas", 0.0)),
                c_et_gap=_clean_small(-_f(et_ph, "neg_et_bt_gap", 0.0)),
                c_et_gap_lag1=_clean_small(-_f(et_ph, "neg_et_bt_gap_lag1", 0.0)),
                c_pressure=_clean_small(-_f(et_ph, "neg_pressure", 0.0)),
                c_pressure_lag1=_clean_small(-_f(et_ph, "neg_pressure_lag1", 0.0)),
                c_pressure_delta_pos=_clean_small(_f(et_ph, "pressure_delta_pos", 0.0)),
                c_ror=_clean_small(-_f(et_ph, "neg_ror", 0.0)),
                c_et_level=_clean_small(-_f(et_ph, "neg_et_level", 0.0)),
            ),
            feature_names=bt_feature_names,
            et_feature_names=et_feature_names,
        )
        phase_models[phase] = model

    params.phase_models = phase_models
    return params

