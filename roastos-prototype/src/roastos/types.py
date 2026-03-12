from dataclasses import dataclass

"""This module defines the core data structures for representing the internal state of the roast, 
control inputs, and coffee context parameters. The RoastState class encapsulates the key state variables 
that describe the current conditions of the roast, such as bean temperature, drum energy, moisture content, 
internal pressure, Maillard and development progress, volatile loss, and structural transformation.
 The Control class represents the control inputs that can be applied to the roasting process, including 
 gas percentage, drum pressure, and drum speed. The BeanContext class captures relevant properties o
 f the coffee beans being roasted, such as density, moisture content, and processing method. 
 These data structures provide a standardized way to represent and manipulate the state of the roast and 
 the associated control inputs throughout the dynamics model, state estimation, feature extraction, 
 and flavor prediction components of the system."""

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
    gas_pct: float
    drum_pressure_pa: float
    drum_speed_pct: float


@dataclass
class BeanContext:
    density: float
    moisture: float
    process: str