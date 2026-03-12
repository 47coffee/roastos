from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

"""This module defines the data schemas for the RoastOS gateway API. The RoastMeasurementFrame class represents 
a single measurement frame containing sensor readings and control inputs at a given timestamp during the roast. 
The RoastRecommendation class represents the recommended control adjustments and predicted flavor attributes 
that the gateway will return in response to an API request. These schemas are designed to facilitate 
structured communication between the roast controller and any external clients or interfaces that interact with the RoastOS system."""

class RoastMeasurementFrame(BaseModel):
    timestamp_s: float
    bt_c: float
    et_c: float
    ror_c_per_min: float
    gas_pct: float = Field(ge=0, le=100)
    drum_pressure_pa: float = Field(ge=0)
    drum_speed_pct: float = Field(ge=0, le=100)
    machine_state: Literal["idle", "preheating", "roasting", "cooling", "stopped"]


class RoastRecommendation(BaseModel):
    recommended_gas_pct: float = Field(ge=0, le=100)
    recommended_drum_pressure_pa: float = Field(ge=0)
    recommended_drum_speed_pct: float = Field(ge=0, le=100)
    message: str
    predicted_clarity: Optional[float] = None
    predicted_sweetness: Optional[float] = None
    predicted_body: Optional[float] = None
    predicted_bitterness: Optional[float] = None