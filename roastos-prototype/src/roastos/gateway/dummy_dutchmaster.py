from __future__ import annotations

import random

from roastos.dynamics import observation_from_state, step_dynamics
from roastos.gateway.base import BaseRoasterGateway
from roastos.gateway.schemas import RoastMeasurementFrame
from roastos.state import initial_state
from roastos.types import Control, RoastState

"""This module defines a dummy gateway implementation for simulating a Dutch Masters roasting machine. 
The DummyDutchMasterGateway class inherits from the BaseRoasterGateway and provides a simple simulation of the 
roasting process using the dynamics model defined in the roastos.dynamics module. The gateway maintains an internal 
state of the roast and updates it based on the control inputs applied through the apply_control method. 
The read_frame method simulates reading sensor data from the machine by advancing the internal state and 
adding some  random noise to the measurements of bean temperature, environment temperature, and rate of rise (RoR). 
This dummy gateway can be used for testing and development purposes before integrating with a
 real roasting machine API. It allows for simulating the behavior of the machine and generating 
 realistic measurement frames that can be used to validate the controller and other components of the RoastOS system."""

class DummyDutchMasterGateway(BaseRoasterGateway):
    """
    Dummy machine gateway that simulates a Dutch Masters roaster.

    RoastOS will later swap this for a real API gateway.
    """

    def __init__(
        self,
        *,
        dt_s: float = 2.0,
        coffee_context: dict | None = None,
        initial_control: Control | None = None,
        noise_std_bt: float = 0.15,
        noise_std_et: float = 0.25,
        noise_std_ror: float = 2.0,
    ):
        self.dt_s = dt_s
        self.coffee_context = coffee_context or {
            "density": 0.78,
            "moisture": 0.11,
        }
        self.state: RoastState = initial_state()
        self.control = initial_control or Control(
            gas_pct=75.0,
            drum_pressure_pa=90.0,
            drum_speed_pct=65.0,
        )
        self.time_s = 0.0
        self.machine_state = "roasting"

        self.noise_std_bt = noise_std_bt
        self.noise_std_et = noise_std_et
        self.noise_std_ror = noise_std_ror

    def connect(self) -> None:
        # Dummy gateway does not need a real connection
        pass

    def apply_control(self, control: Control) -> None:
        self.control = control

    def _advance_hidden_state(self) -> None:
        self.state = step_dynamics(
            self.state,
            self.control,
            coffee_context=self.coffee_context,
            dt_s=self.dt_s,
        )
        self.time_s += self.dt_s

    def read_frame(self) -> RoastMeasurementFrame:
        self._advance_hidden_state()

        bt, et = observation_from_state(self.state, self.control)
        ror_c_per_min = self.state.RoR * 60.0

        bt_meas = bt + random.gauss(0.0, self.noise_std_bt)
        et_meas = et + random.gauss(0.0, self.noise_std_et)
        ror_meas = ror_c_per_min + random.gauss(0.0, self.noise_std_ror)

        return RoastMeasurementFrame(
            timestamp_s=self.time_s,
            bt_c=bt_meas,
            et_c=et_meas,
            ror_c_per_min=ror_meas,
            gas_pct=self.control.gas_pct,
            drum_pressure_pa=self.control.drum_pressure_pa,
            drum_speed_pct=self.control.drum_speed_pct,
            machine_state=self.machine_state,
        )