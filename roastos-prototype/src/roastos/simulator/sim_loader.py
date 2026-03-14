from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .sim_types import (
    PhaseBTParams,
    PhaseETParams,
    PhaseModelParams,
    SimulatorParams,
    PHASES,
)

DEFAULT_MODEL_PATH = "artifacts/models/physics_model_v3_0.json"


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


def load_simulator_params(model_json_path: str | Path = DEFAULT_MODEL_PATH) -> SimulatorParams:
    path = Path(model_json_path)
    payload = _read_json(path)

    phases_payload = payload.get("phases", {})
    et_models_payload = payload.get("et_models", {})
    latent_payload = payload.get("latent_state", {})
    best_config = payload.get("best_config", {})

    if not isinstance(phases_payload, dict):
        raise ValueError("Expected 'phases' in model JSON.")
    if not isinstance(et_models_payload, dict):
        et_models_payload = {}

    params = SimulatorParams()

    params.dt_sec = 1.0
    params.bt_norm_denominator = 200.0
    params.bt_norm_offset = 0.0
    params.et_norm_denominator = 250.0
    params.ror_filter_alpha = 0.70
    params.ror_model_clip = 30.0

    params.gas_already_normalized = True
    params.pressure_is_raw_pa = True

    params.latent_decay = float(best_config.get("decay", latent_payload.get("best_decay", 0.99)))
    params.latent_pressure_scale = float(
        best_config.get("pressure_scale", latent_payload.get("best_pressure_scale", 100.0))
    )
    params.latent_mean = float(latent_payload.get("raw_mean", 0.0))
    params.latent_std = float(latent_payload.get("raw_std", 1.0))
    if abs(params.latent_std) < 1e-9:
        params.latent_std = 1.0

    params.include_gas_feature = bool(best_config.get("include_gas", False))

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