from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from roastos.types import Control, RoastState

"""This module defines the RoastMPC class, which implements a more robust nonlinear Model Predictive Control (MPC) 
strategy for optimizing the control inputs during the roasting process. The MPC formulation includes move blocking, rate constraints on 
the control inputs, and a softer objective function that balances multiple aspects of the roast's internal state with 
the target flavor profile. The optimize method sets up and solves the MPC problem using CasADi, and includes a 
graceful fallback mechanism in case the solver fails to find a solution, ensuring that the system can continue 
operating by holding the current control inputs over the horizon. This class serves as an advanced control 
optimization component within the RoastOS system, enabling it to recommend control actions that are 
expected to yield the desired flavor outcome based on the current state of the roast and the coffee context parameters."""

MIN_PRESSURE_PA = 50.0
MAX_PRESSURE_PA = 150.0


def _pressure_norm_symbolic(pressure_pa):
    return ca.fmax(
        0.0,
        ca.fmin(1.0, (pressure_pa - MIN_PRESSURE_PA) / (MAX_PRESSURE_PA - MIN_PRESSURE_PA)),
    )


@dataclass
class MPCResult:
    controls: list[Control]
    objective_value: float
    success: bool
    status: str


class RoastMPC:
    """
    More robust v1 nonlinear MPC:
    - move blocking
    - rate constraints
    - softer objective
    - graceful fallback on failure
    """

    def __init__(
        self,
        horizon_steps: int = 20,
        dt_s: float = 2.0,
        n_blocks: int = 4,
    ):
        self.N = horizon_steps
        self.dt_s = dt_s
        self.n_blocks = n_blocks
        self.block_len = max(1, horizon_steps // n_blocks)

    def _step_symbolic(self, x, u, density: float, moisture: float):
        Tb = x[0]
        RoR = x[1]
        E_drum = x[2]
        M = x[3]
        P_int = x[4]
        p_mai = x[5]
        p_dev = x[6]
        V_loss = x[7]
        S_struct = x[8]

        gas_pct = u[0]
        pressure_pa = u[1]
        drum_speed_pct = u[2]

        gas = gas_pct / 100.0
        pnorm = _pressure_norm_symbolic(pressure_pa)
        drum_speed = drum_speed_pct / 100.0

        density_factor = 1.0 - 0.15 * (density - 0.78)
        moisture_factor = 1.0 - 0.35 * (moisture - 0.11)
        bean_response = ca.fmax(0.75, ca.fmin(1.25, density_factor * moisture_factor))

        # Drum energy
        E_amb = 0.35
        a_g = 0.070
        a_p = 0.035
        a_l = 0.020
        dE = (a_g * gas - a_p * pnorm - a_l * (E_drum - E_amb)) * (self.dt_s / 2.0)
        E_drum_next = ca.fmax(0.0, ca.fmin(1.6, E_drum + dE))

        # Environment
        T_base = 150.0
        b_d = 55.0
        b_g = 60.0
        b_p = 20.0
        T_env = T_base + b_d * E_drum + b_g * gas - b_p * pnorm

        # Moisture
        T_evap = 100.0
        c_m = 0.010
        evap_gate = 1.0 / (1.0 + ca.exp(-0.10 * (Tb - T_evap)))
        r_evap = c_m * evap_gate * M * (0.9 + 1.2 * moisture) * (1.0 + 0.25 * ca.fmax(RoR, 0.0))
        M_next = ca.fmax(0.0, ca.fmin(0.20, M - r_evap * (self.dt_s / 2.0)))

        # Pressure
        T_p = 160.0
        c_p1 = 0.030
        c_p2 = 0.040
        c_p3 = 0.020
        pressure_build = c_p1 * (1.0 / (1.0 + ca.exp(-0.12 * (Tb - T_p)))) * (M / 0.12)
        pressure_release = c_p2 * P_int + c_p3 * pnorm
        P_int_next = ca.fmax(0.0, ca.fmin(2.0, P_int + (pressure_build - pressure_release) * (self.dt_s / 2.0)))

        # RoR heat balance
        k_h = 0.022 * bean_response
        k_e = 1.8
        k_r = 0.10
        dRoR = (k_h * (T_env - Tb) - k_e * M - k_r * RoR) * (self.dt_s / 2.0)
        dRoR = dRoR - 0.12 * ca.fmax(drum_speed - 0.65, 0.0)
        RoR_next = ca.fmax(-1.0, ca.fmin(5.0, RoR + dRoR))

        # Temperature
        Tb_next = ca.fmax(20.0, ca.fmin(260.0, Tb + RoR_next * self.dt_s))

        # Maillard
        T_mai = 148.0
        c_mai = 0.010
        mai_rate = (
            c_mai
            * (1.0 / (1.0 + ca.exp(-0.12 * (Tb - T_mai))))
            * (1.0 - p_mai)
            * (1.0 - 0.5 * (M / 0.12))
            * ca.exp(0.010 * ca.fmax(Tb - 150.0, 0.0))
        )
        p_mai_next = ca.fmax(0.0, ca.fmin(1.0, p_mai + mai_rate * (self.dt_s / 2.0)))

        # Development
        P_fc = 0.20
        c_dev = 0.020
        dev_rate = c_dev * (1.0 / (1.0 + ca.exp(-15.0 * (P_int - P_fc)))) * (1.0 - p_dev)
        p_dev_next = ca.fmax(0.0, ca.fmin(1.0, p_dev + dev_rate * (self.dt_s / 2.0)))

        # Volatile loss
        T_v0 = 170.0
        c_v = 0.0018
        alpha_v = 0.030
        beta_p = 0.9
        thermal_excess = ca.fmax(Tb - T_v0, 0.0)
        vloss_rate = c_v * ca.exp(alpha_v * thermal_excess) * (1.0 + beta_p * pnorm)
        V_loss_next = ca.fmax(0.0, ca.fmin(3.0, V_loss + vloss_rate * (self.dt_s / 2.0)))

        # Structure
        T_s = 160.0
        c_s1 = 0.040
        c_s2 = 0.090
        c_s3 = 0.004
        struct_rate = (
            c_s1 * p_mai
            + c_s2 * p_dev
            + c_s3 * (1.0 / (1.0 + ca.exp(-0.10 * (Tb - T_s))))
        )
        S_struct_next = ca.fmax(0.0, ca.fmin(3.0, S_struct + struct_rate * (self.dt_s / 2.0)))

        return ca.vertcat(
            Tb_next,
            RoR_next,
            E_drum_next,
            M_next,
            P_int_next,
            p_mai_next,
            p_dev_next,
            V_loss_next,
            S_struct_next,
        )

    def _expand_block_controls(self, U_block):
        """
        Expand block controls [3, n_blocks] into per-step controls [3, N].
        """
        cols = []
        for b in range(self.n_blocks):
            for _ in range(self.block_len):
                cols.append(U_block[:, b])

        # Trim or pad to exactly N
        if len(cols) < self.N:
            while len(cols) < self.N:
                cols.append(U_block[:, self.n_blocks - 1])
        cols = cols[: self.N]
        return ca.hcat(cols)

    def optimize(
        self,
        *,
        x0: RoastState,
        current_control: Control,
        target_structure: dict,
        coffee_context: dict,
    ) -> MPCResult:
        opti = ca.Opti()

        # Optimize only block controls
        U_block = opti.variable(3, self.n_blocks)
        U = self._expand_block_controls(U_block)
        X = opti.variable(9, self.N + 1)

        density = float(coffee_context.get("density", 0.78))
        moisture = float(coffee_context.get("moisture", 0.11))

        x0_vec = ca.vertcat(
            x0.Tb,
            x0.RoR,
            x0.E_drum,
            x0.M,
            x0.P_int,
            x0.p_mai,
            x0.p_dev,
            x0.V_loss,
            x0.S_struct,
        )

        opti.subject_to(X[:, 0] == x0_vec)

        # Bounds
        opti.subject_to(opti.bounded(0.0, U_block[0, :], 100.0))
        opti.subject_to(opti.bounded(MIN_PRESSURE_PA, U_block[1, :], MAX_PRESSURE_PA))
        opti.subject_to(opti.bounded(50.0, U_block[2, :], 80.0))

        # Rate constraints between blocks
        max_dgas = 8.0
        max_dpressure = 15.0
        max_ddrum = 5.0

        prev_gas = current_control.gas_pct
        prev_pressure = current_control.drum_pressure_pa
        prev_drum = current_control.drum_speed_pct

        for b in range(self.n_blocks):
            opti.subject_to(opti.bounded(-max_dgas, U_block[0, b] - prev_gas, max_dgas))
            opti.subject_to(opti.bounded(-max_dpressure, U_block[1, b] - prev_pressure, max_dpressure))
            opti.subject_to(opti.bounded(-max_ddrum, U_block[2, b] - prev_drum, max_ddrum))

            prev_gas = U_block[0, b]
            prev_pressure = U_block[1, b]
            prev_drum = U_block[2, b]

        # Dynamics rollout
        J = 0
        for k in range(self.N):
            x_next = self._step_symbolic(X[:, k], U[:, k], density=density, moisture=moisture)
            opti.subject_to(X[:, k + 1] == x_next)

            # Small running penalties for control effort
            J += 0.002 * (U[0, k] - current_control.gas_pct) ** 2
            J += 0.0005 * (U[1, k] - current_control.drum_pressure_pa) ** 2
            J += 0.0005 * (U[2, k] - current_control.drum_speed_pct) ** 2

        # Terminal objective
        TbN = X[0, -1]
        RoRN = X[1, -1]
        MN = X[3, -1]
        pMaiN = X[5, -1]
        pDevN = X[6, -1]
        vLossN = X[7, -1]
        sStructN = X[8, -1]

        pDryN = 1.0 - (MN / 0.12)

        # Softer weights than before
        J += 1.5 * (pDryN - target_structure["dry"]) ** 2
        J += 1.5 * (pMaiN - target_structure["maillard"]) ** 2
        J += 2.0 * (pDevN - target_structure["dev"]) ** 2
        J += 1.2 * (vLossN - target_structure["volatile_loss"]) ** 2
        J += 1.2 * (sStructN - target_structure["structure"]) ** 2
        J += 0.05 * (RoRN * 60.0 - target_structure["ror_fc"]) ** 2
        J += 0.02 * (TbN - target_structure["Tb_end_c"]) ** 2

        opti.minimize(J)

        # Strong initial guess = hold current control
        opti.set_initial(U_block[0, :], current_control.gas_pct)
        opti.set_initial(U_block[1, :], current_control.drum_pressure_pa)
        opti.set_initial(U_block[2, :], current_control.drum_speed_pct)

        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 300,
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 5,
                "ipopt.mu_strategy": "adaptive",
            },
        )

        try:
            sol = opti.solve()

            controls = []
            U_val = sol.value(U)
            for k in range(self.N):
                controls.append(
                    Control(
                        gas_pct=float(U_val[0, k]),
                        drum_pressure_pa=float(U_val[1, k]),
                        drum_speed_pct=float(U_val[2, k]),
                    )
                )

            return MPCResult(
                controls=controls,
                objective_value=float(sol.value(J)),
                success=True,
                status="Solve_Succeeded",
            )

        except RuntimeError as exc:
            # Graceful fallback: hold current control over the horizon
            fallback_controls = [
                Control(
                    gas_pct=current_control.gas_pct,
                    drum_pressure_pa=current_control.drum_pressure_pa,
                    drum_speed_pct=current_control.drum_speed_pct,
                )
                for _ in range(self.N)
            ]

            return MPCResult(
                controls=fallback_controls,
                objective_value=float("nan"),
                success=False,
                status=f"Fallback_Applied: {str(exc)}",
            )