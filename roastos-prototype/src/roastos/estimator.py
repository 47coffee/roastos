from __future__ import annotations

from roastos.dynamics import observation_from_state, step_dynamics, _pressure_norm
from roastos.gateway.schemas import RoastMeasurementFrame
from roastos.types import Control, RoastState

"""This module defines the RoastStateEstimator class, which implements a lightweight predict-correct observer 
for estimating the internal state of the roast based on sensor measurements. 
The estimator uses the step_dynamics function to predict the next state given 
the current state and control inputs, and then applies a correction based on the 
residuals between the predicted sensor measurements (bean temperature and environment temperature) 
and the actual measurements from the RoastMeasurementFrame. The update method adjusts the internal 
state variables such as bean temperature, rate of rise (RoR), drum energy, and internal pressure using
simple proportional gains on the residuals. This estimator serves as a bridge toward implementing a full Extended Kalman Filter (EKF) 
for more sophisticated state estimation in future iterations of the RoastOS system."""

class RoastStateEstimator:
    """
    Lightweight predict-correct observer.
    This is not yet a full EKF, but it is the right bridge toward one.
    """

    def __init__(
        self,
        initial_state: RoastState,
        *,
        dt_s: float = 2.0,
        coffee_context: dict | None = None,
    ):
        self.state = initial_state
        self.dt_s = dt_s
        self.coffee_context = coffee_context or {"density": 0.78, "moisture": 0.11}

    def predict(self, control: Control) -> RoastState:
        self.state = step_dynamics(
            self.state,
            control,
            coffee_context=self.coffee_context,
            dt_s=self.dt_s,
        )
        return self.state

    def update(self, frame: RoastMeasurementFrame, control: Control) -> RoastState:
        """
        Correct the predicted state using sensor measurements.
        """
        bt_pred, et_pred = observation_from_state(self.state, control)

        bt_resid = frame.bt_c - bt_pred
        et_resid = frame.et_c - et_pred
        ror_resid = (frame.ror_c_per_min / 60.0) - self.state.RoR

        # Basic correction gains
        k_bt = 0.55
        k_ror = 0.35
        k_et_to_edrum = 0.015

        Tb = self.state.Tb + k_bt * bt_resid
        RoR = self.state.RoR + k_ror * ror_resid

        # Use ET residual to nudge drum energy
        E_drum = self.state.E_drum + k_et_to_edrum * et_resid

        # Slight pressure correction from BT + pressure operating point
        pnorm = _pressure_norm(frame.drum_pressure_pa)
        P_int = self.state.P_int + 0.01 * bt_resid - 0.003 * pnorm

        self.state = RoastState(
            Tb=Tb,
            RoR=RoR,
            E_drum=max(0.0, min(1.6, E_drum)),
            M=self.state.M,
            P_int=max(0.0, min(2.0, P_int)),
            p_mai=self.state.p_mai,
            p_dev=self.state.p_dev,
            V_loss=self.state.V_loss,
            S_struct=self.state.S_struct,
        )
        return self.state