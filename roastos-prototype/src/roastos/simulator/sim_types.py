from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PHASES = ("drying", "maillard", "development")


@dataclass(frozen=True)
class RoastContext:
    roast_id: Optional[str] = None
    start_weight_kg: Optional[float] = None

    # new explicit fields
    bean_start_temp_c: Optional[float] = None
    charge_temp_c: Optional[float] = None

    # backward compatibility alias only
    start_temp_c: Optional[float] = None

    target_drop_bt: Optional[float] = None
    target_drop_weight_kg: Optional[float] = None

    def __post_init__(self):
        if self.bean_start_temp_c is None and self.start_temp_c is not None:
            object.__setattr__(self, "bean_start_temp_c", self.start_temp_c)

    @property
    def effective_bean_start_temp_c(self) -> Optional[float]:
        return self.bean_start_temp_c

    @property
    def effective_charge_temp_c(self) -> Optional[float]:
        return self.charge_temp_c


@dataclass
class RoastControl:
    gas: float
    pressure: float
    drum_speed: float = 0.65


@dataclass
class RoastSimState:
    t_sec: float
    bt: float
    et: float
    ror: float

    e_drum_raw: float = 0.0
    e_drum: float = 0.0

    m_burden: float = 1.0
    p_dry: float = 0.0
    p_mai: float = 0.0
    p_dev: float = 0.0

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
            m_burden=self.m_burden,
            p_dry=self.p_dry,
            p_mai=self.p_mai,
            p_dev=self.p_dev,
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
    intercept: float = 0.0
    c_e_drum: float = 0.0
    c_et_delta: float = 0.0
    c_bt_level: float = 0.0
    c_ror: float = 0.0
    c_pressure_direct: float = 0.0
    c_gas: float = 0.0


@dataclass
class PhaseETParams:
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
    bt: PhaseBTParams = field(default_factory=PhaseBTParams)
    et: PhaseETParams = field(default_factory=PhaseETParams)
    feature_names: List[str] = field(default_factory=list)
    et_feature_names: List[str] = field(default_factory=list)


@dataclass
class TerminalOutputs:
    drop_bt: float
    drop_time_s: float
    drop_weight_kg: Optional[float] = None
    loss_fraction: Optional[float] = None
    dtr: Optional[float] = None


@dataclass
class SimulatorParams:
    dt_sec: float = 1.0

    bt_norm_offset: float = 0.0
    bt_norm_denominator: float = 200.0
    et_norm_denominator: float = 250.0

    ror_filter_alpha: float = 0.70
    ror_model_clip: float = 30.0

    gas_already_normalized: bool = True
    pressure_is_raw_pa: bool = True

    latent_decay: float = 0.99
    latent_pressure_scale: float = 100.0
    latent_mean: float = 0.0
    latent_std: float = 1.0

    include_gas_feature: bool = False

    moisture_decay_rate: float = 0.015
    moisture_heat_coeff: float = 0.002
    moisture_bt_drag_coeff: float = 0.0
    moisture_et_drag_coeff: float = 0.0

    progress_drying_bt_start: float = 100.0
    progress_drying_bt_end: float = 160.0
    progress_maillard_bt_start: float = 150.0
    progress_maillard_bt_end: float = 196.0
    progress_development_bt_start: float = 196.0
    progress_development_bt_end: float = 222.0

    enable_context_dynamics: bool = False
    reference_charge_weight_kg: float = 6.0

    # new explicit fields
    reference_bean_start_temp_c: float = 25.0
    reference_charge_temp_c: float = 230.0

    # backward compatibility only
    reference_start_temp_c: float = 25.0

    max_context_response_scale: float = 1.35
    min_context_response_scale: float = 0.70

    drop_weight_moisture_coeff: float = 0.12
    drop_weight_development_coeff: float = 0.05
    drop_weight_maillard_coeff: float = 0.02

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

