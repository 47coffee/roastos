import numpy as np

"""This module defines the state estimation logic for inferring the internal state of the roast from observed features."""

class RoastStateEstimator:

    def __init__(self, state_dim, obs_dim):

        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 0.1

        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(obs_dim) * 0.5

    def predict(self, f, u):

        x_pred = f(self.x, u)

        F = np.eye(self.state_dim)   # simplified Jacobian

        P_pred = F @ self.P @ F.T + self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, z, h):

        z_pred = h(self.x)

        H = np.zeros((self.obs_dim, self.state_dim))

        H[0,0] = 1
        H[1,0] = 0.5

        y = z - z_pred

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P