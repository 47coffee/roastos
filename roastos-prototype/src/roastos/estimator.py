from __future__ import annotations

import numpy as np

from roastos.types import RoastState, Control
from roastos.dynamics import step_dynamics, observation_from_state


class RoastStateEstimator:
    """
    Lightweight EKF-style state estimator for RoastOS.
    """

    def __init__(self, initial_state: RoastState):
        self.state = initial_state

        # covariance
        self.P = np.eye(10) * 0.05

        # process noise
        self.Q = np.eye(10) * 0.01

        # measurement noise
        self.R = np.eye(2) * 0.5

    def predict(self, control: Control, coffee_context: dict, dt_s: float):
        self.state = step_dynamics(
            self.state,
            control,
            coffee_context,
            dt_s,
        )

        # simplified covariance prediction
        self.P = self.P + self.Q

    def update(self, bt_meas, et_meas, control: Control):
        z = np.array([bt_meas, et_meas], dtype=float)

        obs = observation_from_state(self.state, control)
        h = np.array(obs, dtype=float)

        # Observation model:
        # BT = Tb
        # ET = Tb + 55 * E_drum
        H = np.zeros((2, 10))
        H[0, 0] = 1.0
        H[1, 0] = 1.0
        H[1, 2] = 32.0

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = z - h
        dx = K @ innovation

        x = np.array([
            self.state.Tb,
            self.state.RoR,
            self.state.E_drum,
            self.state.M,
            self.state.P_int,
            self.state.p_mai,
            self.state.p_dev,
            self.state.V_loss,
            self.state.S_struct,
            self.state.Q_bias,
        ], dtype=float)

        x = x + dx

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        self.state = RoastState(
            Tb=x[0],
            RoR=clamp(x[1], -0.5, 0.5),        # about -30 to +30 °C/min
            E_drum=clamp(x[2], 0.0, 1.0),
            M=clamp(x[3], 0.0, 0.12),
            P_int=clamp(x[4], 0.0, 2.0),
            p_mai=clamp(x[5], 0.0, 1.0),
            p_dev=clamp(x[6], 0.0, 1.0),
            V_loss=clamp(x[7], 0.0, 1.0),
            S_struct=clamp(x[8], 0.0, 1.0),
            Q_bias=clamp(x[9], -1.0, 1.0),    # bounded bias term for model mismatch
        )

        I = np.eye(10)
        self.P = (I - K @ H) @ self.P

        return self.state