from dataclasses import dataclass

"""This module defines core data types used across the RoastOS codebase, including representations of roast state, control inputs, and bean context.
These types are used for modeling the roasting process and building datasets for machine learning."""

@dataclass
class RoastState:
    Tb: float
    RoR: float
    E_drum: float
    M: float
    P_int: float
    p_mai: float
    p_dev: float
    V_loss: float
    S_struct: float

@dataclass
class Control:
    gas: float
    airflow: float
    drum_speed: float


@dataclass
class BeanContext:
    density: float
    moisture: float
    process: str