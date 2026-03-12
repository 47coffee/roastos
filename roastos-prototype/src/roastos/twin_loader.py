from __future__ import annotations

import json
from pathlib import Path


DEFAULT_TWIN_PARAMS = {
    "alpha_gas": 0.040,
    "beta_et": 0.018,
    "gamma_pressure": 0.003,
    "delta_ror": 0.012,
    "intercept": 0.0,
    "ror_gas_gain": 0.010,
    "ror_et_gain": 0.006,
    "ror_pressure_cooling": 0.004,
    "ror_progress_decay": 0.020,
    "moisture_evap_coeff": 0.0025,
    "pressure_build_coeff": 0.0015,
    "pressure_release_coeff": 0.0040,
}


def load_twin_params(path: str | Path = "artifacts/models/physics_model.json") -> dict:
    path = Path(path)

    if not path.exists():
        return DEFAULT_TWIN_PARAMS.copy()

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    params = DEFAULT_TWIN_PARAMS.copy()
    params.update(loaded)
    return params