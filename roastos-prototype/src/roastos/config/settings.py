from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    processed_folder: Path
    calibration_dataset: Path
    model_artifact: Path
    replay_output_dir: Path


@dataclass(frozen=True)
class SimulatorConfig:
    dt_sec: float = 1.0
    bt_norm_offset: float = 0.0
    bt_norm_denominator: float = 200.0
    et_norm_denominator: float = 250.0
    ror_filter_alpha: float = 0.70
    ror_model_clip: float = 30.0
    gas_already_normalized: bool = True
    pressure_is_raw_pa: bool = True


@dataclass(frozen=True)
class PhaseThresholdsConfig:
    drying_end_bt: float = 160.0
    maillard_end_bt: float = 196.0


@dataclass(frozen=True)
class ReplayConfig:
    teacher_force_et: bool = False
    teacher_force_ror: bool = False
    teacher_force_phase: bool = True
    default_roast_ids: tuple[str, ...] = ("PR-0173", "PR-0181", "PR-0182", "PR-0186", "PR-0180")
    warmup_rows: int = 120


@dataclass(frozen=True)
class CalibrationConfig:
    bt_model_version: str = "v3.0"
    et_model_version: str = "et_v3.0"
    release_label: str = "V3.1"
    release_notes: str = (
        "V3.1 replay robustness + config layer + roast-start alignment. "
        "V3.0 thermal plant remains the frozen historical baseline artifact. "
        "Context variables are carried through dataset/runtime interfaces."
    )
    phase_names: tuple[str, ...] = ("drying", "maillard", "development")
    latent_decay_grid: tuple[float, ...] = (0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99)
    pressure_scale_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    include_gas_options: tuple[bool, ...] = (False, True)


@dataclass(frozen=True)
class StateModelConfig:
    canonical_state_names: tuple[str, ...] = (
        "bt",
        "et",
        "ror",
        "e_drum",
        "m_burden",
        "p_dry",
        "p_mai",
        "p_dev",
    )
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


@dataclass(frozen=True)
class ContextModelConfig:
    enable_context_dynamics: bool = False
    reference_charge_weight_kg: float = 6.0

    # New explicit temperature definitions
    reference_bean_start_temp_c: float = 25.0
    default_bean_start_temp_c: float | None = 25.0

    reference_charge_temp_c: float = 230.0
    default_charge_temp_c: float | None = 230.0

    # Backward compatibility only
    reference_start_temp_c: float | None = 25.0
    default_start_temp_c: float | None = 25.0

    max_context_response_scale: float = 1.35
    min_context_response_scale: float = 0.70
    default_start_weight_kg: float | None = None

    drop_weight_moisture_coeff: float = 0.12
    drop_weight_development_coeff: float = 0.05
    drop_weight_maillard_coeff: float = 0.02


@dataclass(frozen=True)
class EKFConfig:
    enabled: bool = True
    k_bt: float = 0.35
    k_et: float = 0.25
    k_ror: float = 0.20


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 20
    move_block: int = 5
    gas_min: float = 0.0
    gas_max: float = 1.0
    pressure_min: float = 0.0
    pressure_max: float = 240.0
    gas_move_penalty: float = 2.0
    pressure_move_penalty: float = 0.02
    bt_track_weight: float = 1.0
    et_track_weight: float = 0.35
    terminal_bt_weight: float = 1.25
    terminal_et_weight: float = 0.50
    terminal_drop_weight_weight: float = 1.0


@dataclass(frozen=True)
class FlavourConfig:
    enabled: bool = False
    predictor_mode: str = "stub"
    default_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sweetness": 1.0,
            "clarity": 1.0,
            "body": 1.0,
            "acidity": 1.0,
            "bitterness": 1.0,
            "aroma": 1.0,
        }
    )


@dataclass(frozen=True)
class RoastOSSettings:
    paths: PathsConfig
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    phase_thresholds: PhaseThresholdsConfig = field(default_factory=PhaseThresholdsConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    state_model: StateModelConfig = field(default_factory=StateModelConfig)
    context_model: ContextModelConfig = field(default_factory=ContextModelConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    flavour: FlavourConfig = field(default_factory=FlavourConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

