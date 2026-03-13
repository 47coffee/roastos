from pathlib import Path
import json


MODEL_PATH = Path("artifacts/models/physics_model.json")


# ------------------------------------------------------------
# physics priors (stable defaults)
# ------------------------------------------------------------

PHYSICS_PRIORS = {

    "intercept": 0.12,
    "alpha_gas": 0.8,
    "beta_et": 0.006,
    "gamma_pressure": 0.002,
    "delta_ror": 0.4,

    "moisture_evap_coeff": 0.00025,
    "pressure_build_coeff": 0.00018,
    "pressure_release_coeff": 0.00040
}


# blending weight
CALIBRATION_WEIGHT = 0.30


def load_twin_params(path: str | Path = MODEL_PATH):

    path = Path(path)

    if not path.exists():
        print("Physics model not found — using physics priors")
        return PHYSICS_PRIORS.copy()

    with open(path, "r") as f:
        calibrated = json.load(f)

    params = {}

    for key, default_val in PHYSICS_PRIORS.items():

        calib_val = calibrated.get(key, default_val)

        params[key] = (
            (1 - CALIBRATION_WEIGHT) * default_val
            + CALIBRATION_WEIGHT * calib_val
        )

    return params