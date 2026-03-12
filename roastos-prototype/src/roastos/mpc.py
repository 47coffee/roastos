from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from roastos.types import Control, RoastState
from roastos.twin_loader import load_twin_params


MIN_PRESSURE_PA = 50.0
MAX_PRESSURE_PA = 150.0


@dataclass
class MPCResult:
    controls: list[Control]
    objective_value: float
    success: bool
    status: str


class RoastMPC:
    """
    RoastOS nonlinear MPC using the calibrated Digital Twin model.

    Notes:
    - Uses move blocking for robustness
    - Optimizes toward structural targets for now
    - Falls back gracefully if IPOPT fails
    """

    def __init__(
        self,
        horizon_steps: int = 20,
        dt_s: float = 2.0,
        n_blocks: int = 4,
        physics_model_path: str = "artifacts/models/physics_model.json",
    ):
        self.N = horizon_steps
        self.dt_s = dt_s
        self.n_blocks = n_blocks
        self.block_len = max(1, horizon_steps // n_blocks)
        self.params = load_twin_params(physics_model_path)

    def _expand_block_controls(self, U_block):
        """
        Expand block controls [3, n_blocks] into per-step controls [3, N].
        """
        cols = []
        for b in range(self.n_blocks):
            for _ in range(self.block_len):
                cols.append(U_block[:, b])

        while len(cols) < self.N:
            cols.append(U_block[:, self.n_blocks - 1])

        cols = cols[: self.N]
        return ca.hcat(cols)

    def _compute_roast_progress(self, M, p_mai, p_dev, moisture0):
        p_dry = 1.0 - (M / moisture0)
        p_dry = ca.fmax(0.0, ca.fmin(1.0, p_dry))

        roast_progress = (
            0.45 * p_dry
            + 0.40 * p_mai
            + 0.15 * p_dev
        )
        return ca.fmax(0.0, ca.fmin(1.0, roast_progress))

    def _step_symbolic(self, x, u, density: float, moisture0: float):
        """
        Symbolic Digital Twin step for CasADi.
        State:
            [Tb, RoR, E_drum, M, P_int, p_mai, p_dev, V_loss, S_struct]
        Control:
            [gas_pct, drum_pressure_pa, drum_speed_pct]
        """
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
        pressure = pressure_pa
        drum_speed = drum_speed_pct / 100.0

        roast_progress = self._compute_roast_progress(M, p_mai, p_dev, moisture0)

        # ------------------------------------------------------------
        # Drum energy
        # ------------------------------------------------------------
        dE = (
            0.018 * gas
            - (0.010 + 0.006 * (pressure / 120.0)) * E_drum
        ) * self.dt_s

        E_drum_next = ca.fmax(0.0, ca.fmin(1.0, E_drum + dE))

        # ------------------------------------------------------------
        # Environment / ET proxy
        # ------------------------------------------------------------
        ET_proxy = Tb + 170.0 * E_drum_next
        et_delta = ET_proxy - Tb

        # ------------------------------------------------------------
        # BT update from calibrated coefficients
        # IMPORTANT:
        # physics_calibration used pressure in raw Pa units.
        # ------------------------------------------------------------
        dTb = (
            self.params["intercept"]
            + self.params["alpha_gas"] * gas
            + self.params["beta_et"] * et_delta
            - self.params["gamma_pressure"] * pressure
            - self.params["delta_ror"] * (RoR * 60.0)
        )

        Tb_next = Tb + dTb * self.dt_s

        # ------------------------------------------------------------
        # RoR update
        # ------------------------------------------------------------
        dRoR = (
            self.params["ror_gas_gain"] * gas
            + self.params["ror_et_gain"] * et_delta
            - self.params["ror_pressure_cooling"] * pressure
            - self.params["ror_progress_decay"] * roast_progress
            - 0.12 * RoR
        )

        dRoR = dRoR - 0.04 * ca.fmax(drum_speed - 0.65, 0.0)

        RoR_next = ca.fmax(-0.2, ca.fmin(0.8, RoR + dRoR * self.dt_s))

        # ------------------------------------------------------------
        # Moisture
        # ------------------------------------------------------------
        evap = (
            self.params["moisture_evap_coeff"]
            * ca.fmax(0.0, Tb_next - 140.0)
            * M
            * self.dt_s
        )
        M_next = ca.fmax(0.0, M - evap)

        # ------------------------------------------------------------
        # Internal pressure
        # ------------------------------------------------------------
        pressure_build = self.params["pressure_build_coeff"] * ca.fmax(0.0, Tb_next - 170.0)
        pressure_release = self.params["pressure_release_coeff"] * (pressure / 100.0)
        P_int_next = ca.fmax(0.0, P_int + (pressure_build - pressure_release) * self.dt_s)

        # ------------------------------------------------------------
        # Maillard
        # ------------------------------------------------------------
        p_mai_next = ca.fmax(
            0.0,
            ca.fmin(1.0, p_mai + 0.0020 * ca.fmax(0.0, Tb_next - 150.0) * self.dt_s),
        )

        # ------------------------------------------------------------
        # Development
        # ------------------------------------------------------------
        p_dev_next = ca.fmax(
            0.0,
            ca.fmin(1.0, p_dev + 0.0022 * ca.fmax(0.0, Tb_next - 195.0) * self.dt_s),
        )

        # ------------------------------------------------------------
        # Volatile loss
        # ------------------------------------------------------------
        V_loss_next = ca.fmax(
            0.0,
            ca.fmin(1.0, V_loss + 0.0012 * ca.fmax(0.0, Tb_next - 180.0) * self.dt_s),
        )

        # ------------------------------------------------------------
        # Structure
        # ------------------------------------------------------------
        S_struct_next = ca.fmax(
            0.0,
            ca.fmin(
                1.0,
                S_struct + (
                    0.012 * p_dev_next
                    + 0.004 * Tb_next / 200.0
                ) * self.dt_s,
            ),
        )

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

    def optimize(
        self,
        *,
        x0: RoastState,
        current_control: Control,
        target_structure: dict,
        coffee_context: dict,
    ) -> MPCResult:
        opti = ca.Opti()

        U_block = opti.variable(3, self.n_blocks)
        U = self._expand_block_controls(U_block)
        X = opti.variable(9, self.N + 1)

        density = float(coffee_context.get("density", 0.78))
        moisture0 = float(coffee_context.get("moisture", 0.11))

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

        # ------------------------------------------------------------
        # Bounds
        # ------------------------------------------------------------
        opti.subject_to(opti.bounded(0.0, U_block[0, :], 100.0))                  # gas %
        opti.subject_to(opti.bounded(MIN_PRESSURE_PA, U_block[1, :], MAX_PRESSURE_PA))  # pressure Pa
        opti.subject_to(opti.bounded(50.0, U_block[2, :], 80.0))                 # drum speed %

        # ------------------------------------------------------------
        # Rate constraints between blocks
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Dynamics rollout + running cost
        # ------------------------------------------------------------
        J = 0

        for k in range(self.N):
            x_next = self._step_symbolic(X[:, k], U[:, k], density=density, moisture0=moisture0)
            opti.subject_to(X[:, k + 1] == x_next)

            # small running penalties for control effort
            J += 0.002 * (U[0, k] - current_control.gas_pct) ** 2
            J += 0.0005 * (U[1, k] - current_control.drum_pressure_pa) ** 2
            J += 0.0005 * (U[2, k] - current_control.drum_speed_pct) ** 2

        # ------------------------------------------------------------
        # Terminal objective (structure-based)
        # ------------------------------------------------------------
        TbN = X[0, -1]
        RoRN = X[1, -1]
        MN = X[3, -1]
        pMaiN = X[5, -1]
        pDevN = X[6, -1]
        vLossN = X[7, -1]
        sStructN = X[8, -1]

        pDryN = 1.0 - (MN / moisture0)
        pDryN = ca.fmax(0.0, ca.fmin(1.0, pDryN))

        # softer terminal weights
        J += 1.5 * (pDryN - target_structure["dry"]) ** 2
        J += 1.5 * (pMaiN - target_structure["maillard"]) ** 2
        J += 2.0 * (pDevN - target_structure["dev"]) ** 2
        J += 1.2 * (vLossN - target_structure["volatile_loss"]) ** 2
        J += 1.2 * (sStructN - target_structure["structure"]) ** 2
        J += 0.05 * ((RoRN * 60.0) - target_structure["ror_fc"]) ** 2
        J += 0.02 * (TbN - target_structure["Tb_end_c"]) ** 2

        opti.minimize(J)

        # ------------------------------------------------------------
        # Initial guess
        # ------------------------------------------------------------
        opti.set_initial(U_block[0, :], current_control.gas_pct)
        opti.set_initial(U_block[1, :], current_control.drum_pressure_pa)
        opti.set_initial(U_block[2, :], current_control.drum_speed_pct)

        # ------------------------------------------------------------
        # Solver config
        # ------------------------------------------------------------
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