from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PHASES = ("drying", "maillard", "development")


@dataclass
class RoastControl:
    """
    Runtime control inputs.

    Assumed calibration-aligned convention:
    - gas is already normalized to 0..1
    - pressure is raw machine airflow / drum pressure units
    - drum_speed is optional and not currently used by the calibrated models
    """
    gas: float
    pressure: float
    drum_speed: float = 0.65


@dataclass
class RoastSimState:
    """
    Main simulator state.

    Notes:
    - bt and et are in degC
    - ror is in degC/min
    - e_drum_raw is the latent unstandardized state
    - e_drum is the standardized latent state used by calibration
    - bt_prev / et_prev / prev_pressure are carried to reproduce lagged ET-model features
    """
    t_sec: float
    bt: float
    et: float
    ror: float

    e_drum_raw: float = 0.0
    e_drum: float = 0.0

    phase: str = "drying"

    gas: float = 0.0
    pressure: float = 0.0
    drum_speed: float = 0.65

    bt_prev: Optional[float] = None
    et_prev: Optional[float] = None
    prev_pressure: Optional[float] = None

    def copy(self) -> "RoastSimState":
        return RoastSimState(
            t_sec=self.t_sec,
            bt=self.bt,
            et=self.et,
            ror=self.ror,
            e_drum_raw=self.e_drum_raw,
            e_drum=self.e_drum,
            phase=self.phase,
            gas=self.gas,
            pressure=self.pressure,
            drum_speed=self.drum_speed,
            bt_prev=self.bt_prev,
            et_prev=self.et_prev,
            prev_pressure=self.prev_pressure,
        )


@dataclass
class PhaseBTParams:
    """
    Phase-specific calibrated BT-delta coefficients.

    Simulator-native mapping:
    - intercept            -> intercept
    - e_drum              -> c_e_drum
    - et_delta            -> c_et_delta
    - neg_bt_level        -> c_bt_level = -coef
    - neg_ror             -> c_ror = -coef
    - neg_pressure_direct -> c_pressure_direct = -coef
    - gas                 -> c_gas
    """
    intercept: float = 0.0
    c_e_drum: float = 0.0
    c_et_delta: float = 0.0
    c_bt_level: float = 0.0
    c_ror: float = 0.0
    c_pressure_direct: float = 0.0
    c_gas: float = 0.0


@dataclass
class PhaseETParams:
    """
    Phase-specific calibrated ET-step coefficients.

    Target:
        et_step = et_next - et

    Simulator-native mapping:
    - intercept            -> intercept
    - e_drum               -> c_e_drum
    - gas                  -> c_gas
    - neg_et_bt_gap        -> c_et_gap = -coef
    - neg_et_bt_gap_lag1   -> c_et_gap_lag1 = -coef
    - neg_pressure         -> c_pressure = -coef
    - neg_pressure_lag1    -> c_pressure_lag1 = -coef
    - pressure_delta_pos   -> c_pressure_delta_pos
    - neg_ror              -> c_ror = -coef
    - neg_et_level         -> c_et_level = -coef
    """
    intercept: float = 0.0
    c_e_drum: float = 0.0
    c_gas: float = 0.0
    c_et_gap: float = 0.0
    c_et_gap_lag1: float = 0.0
    c_pressure: float = 0.0
    c_pressure_lag1: float = 0.0
    c_pressure_delta_pos: float = 0.0
    c_ror: float = 0.0
    c_et_level: float = 0.0


@dataclass
class PhaseModelParams:
    """
    Container for all phase-specific model information.
    """
    bt: PhaseBTParams = field(default_factory=PhaseBTParams)
    et: PhaseETParams = field(default_factory=PhaseETParams)
    feature_names: List[str] = field(default_factory=list)
    et_feature_names: List[str] = field(default_factory=list)


@dataclass
class SimulatorParams:
    """
    Calibration-aligned simulator settings.
    """
    dt_sec: float = 1.0

    # bt_c_norm = (bt - bt_norm_offset) / bt_norm_denominator
    # Current calibration contract: bt / 200.0
    bt_norm_offset: float = 0.0
    bt_norm_denominator: float = 200.0

    # et_c_norm = et / 250.0
    et_norm_denominator: float = 250.0

    # RoR handling
    ror_filter_alpha: float = 0.70
    ror_model_clip: float = 30.0

    # Input conventions
    gas_already_normalized: bool = True
    pressure_is_raw_pa: bool = True

    # Latent e_drum reconstruction:
    # e_drum_raw_t = decay * e_drum_raw_t-1 + gas - pressure / pressure_scale
    # e_drum = (e_drum_raw - latent_mean) / latent_std
    latent_decay: float = 0.99
    latent_pressure_scale: float = 100.0
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Whether the fitted BT phase model included gas as direct feature
    include_gas_feature: bool = False

    phase_models: Dict[str, PhaseModelParams] = field(default_factory=dict)


@dataclass
class SimStepResult:
    prev_state: RoastSimState
    control: RoastControl
    next_state: RoastSimState


@dataclass
class ReplayMetrics:
    n_steps: int
    bt_rmse: float
    et_rmse: float
    ror_rmse: float
    bt_mae: float
    et_mae: float
    ror_mae: float
    terminal_bt_error: float
    terminal_et_error: float
    terminal_ror_error: float


@dataclass
class ReplayResult:
    rows: List[dict]
    metrics: ReplayMetrics