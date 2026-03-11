from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

"""This module defines the core data models for the RoastOS system using Pydantic for validation and type enforcement.  
"""


class RoastSession(BaseModel):
    roast_id: str
    timestamp_start: datetime
    machine_id: str
    coffee_id: str
    operator_id: str
    batch_size_kg: float
    style_profile: str
    intent_clarity: float
    intent_sweetness: float
    intent_body: float
    intent_bitterness: float
    charge_temp_c: float
    drop_temp_c: float
    duration_s: int
    ambient_temp_c: float
    ambient_rh_pct: float
    notes: Optional[str] = None

    @field_validator(
        "intent_clarity",
        "intent_sweetness",
        "intent_body",
        "intent_bitterness",
    )
    @classmethod
    def validate_intent_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Intent values must be between 0.0 and 1.0")
        return value

    @field_validator("batch_size_kg", "duration_s")
    @classmethod
    def validate_positive(cls, value: float | int) -> float | int:
        if value <= 0:
            raise ValueError("Value must be positive")
        return value

    @field_validator("ambient_rh_pct")
    @classmethod
    def validate_rh(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("Relative humidity must be between 0 and 100")
        return value


class RoastTimeSeriesRow(BaseModel):
    roast_id: str
    t_s: int
    bt_c: float
    et_c: float
    ror_c_per_min: float
    gas_pct: float
    airflow_pct: float
    drum_speed_pct: float
    x_tb_c: float
    x_edrum: float
    x_pdry: float
    x_pmai: float
    x_pdev: float
    x_vloss: float
    x_sstruct: float
    event_yellow: int
    event_fc_start: int
    event_drop: int

    @field_validator("t_s")
    @classmethod
    def validate_time(cls, value: int) -> int:
        if value < 0:
            raise ValueError("t_s must be >= 0")
        return value

    @field_validator("gas_pct", "airflow_pct", "drum_speed_pct")
    @classmethod
    def validate_percent_controls(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("Control percentages must be between 0 and 100")
        return value

    @field_validator("x_pdry", "x_pmai", "x_pdev")
    @classmethod
    def validate_progress_vars(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Progress variables must be between 0.0 and 1.0")
        return value

    @field_validator("event_yellow", "event_fc_start", "event_drop")
    @classmethod
    def validate_binary_event(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("Event flags must be 0 or 1")
        return value


class RoastFeatures(BaseModel):
    roast_id: str
    pct_dry: float
    pct_maillard: float
    pct_dev: float
    ror_fc: float
    v_loss_final: float
    s_struct_final: float
    crash_index: float
    flick_index: float
    time_to_yellow_s: int
    time_to_fc_s: int
    dev_time_s: int
    delta_bt_fc_to_drop_c: float

    @field_validator("pct_dry", "pct_maillard", "pct_dev")
    @classmethod
    def validate_phase_fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Phase fractions must be between 0.0 and 1.0")
        return value

    @field_validator("v_loss_final", "s_struct_final", "crash_index", "flick_index")
    @classmethod
    def validate_nonnegative_metrics(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("Metric must be non-negative")
        return value

    @field_validator("time_to_yellow_s", "time_to_fc_s", "dev_time_s")
    @classmethod
    def validate_nonnegative_times(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Time values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_phase_sum(self) -> "RoastFeatures":
        total = self.pct_dry + self.pct_maillard + self.pct_dev
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Phase fractions should sum approximately to 1.0, got {total:.4f}"
            )
        return self


class RoastOutcome(BaseModel):
    roast_id: str
    rest_days: int
    brew_method: Literal["cupping", "filter", "espresso"]
    clarity: float
    sweetness: float
    body: float
    bitterness: float
    acidity: float
    aroma: float
    overall_score: float
    panel_size: int
    notes: Optional[str] = None

    @field_validator(
        "clarity",
        "sweetness",
        "body",
        "bitterness",
        "acidity",
        "aroma",
        "overall_score",
    )
    @classmethod
    def validate_scores(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Scores must be between 0.0 and 1.0")
        return value

    @field_validator("rest_days", "panel_size")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be positive")
        return value


class CoffeeLot(BaseModel):
    coffee_id: str
    origin: str
    process: Literal["washed", "natural", "honey", "wet_hulled", "anaerobic", "other"]
    variety: str
    density: float
    moisture: float
    water_activity: float
    screen_size: float
    altitude_m: int

    @field_validator("density", "moisture", "water_activity", "screen_size")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("Value must be positive")
        return value

    @field_validator("altitude_m")
    @classmethod
    def validate_altitude(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Altitude must be positive")
        return value