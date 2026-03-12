from __future__ import annotations

from pathlib import Path
import csv

from roastos.gateway.schemas import RoastMeasurementFrame, RoastRecommendation
from roastos.types import Control, RoastState

"""This module defines the RoastRuntimeLogger class, which provides functionality for logging the roasting 
process in a structured CSV format. The logger captures detailed information about each step of the roasting process, 
including the raw sensor measurements from the machine, the estimated internal state of the roast, the current 
control inputs, the recommended control adjustments, the predicted flavor attributes, and the results of 
the MPC optimization. The log_step method is called at each iteration of the control loop 
to append a new row to the CSV file with all relevant data for that time step. This logging 
capability is essential for analyzing the performance of the RoastOS system, debugging issues, 
and improving future iterations of the control algorithms based on historical data from real or simulated roasts."""

class RoastRuntimeLogger:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def _ensure_header(self) -> None:
        if self._initialized:
            return

        if not self.output_path.exists():
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp_s",
                        "machine_state",
                        "measured_bt_c",
                        "measured_et_c",
                        "measured_ror_c_per_min",
                        "measured_gas_pct",
                        "measured_drum_pressure_pa",
                        "measured_drum_speed_pct",
                        "estimated_Tb",
                        "estimated_RoR_c_per_min",
                        "estimated_E_drum",
                        "estimated_M",
                        "estimated_P_int",
                        "estimated_p_mai",
                        "estimated_p_dev",
                        "estimated_V_loss",
                        "estimated_S_struct",
                        "current_gas_pct",
                        "current_drum_pressure_pa",
                        "current_drum_speed_pct",
                        "recommended_gas_pct",
                        "recommended_drum_pressure_pa",
                        "recommended_drum_speed_pct",
                        "predicted_clarity",
                        "predicted_sweetness",
                        "predicted_body",
                        "predicted_bitterness",
                        "mpc_success",
                        "mpc_objective",
                        "mpc_status",
                        "message",
                    ]
                )

        self._initialized = True

    def log_step(
        self,
        *,
        frame: RoastMeasurementFrame,
        estimated_state: RoastState,
        current_control: Control,
        recommendation: RoastRecommendation,
        mpc_success: bool,
        mpc_objective: float,
        mpc_status: str,
    ) -> None:
        self._ensure_header()

        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    frame.timestamp_s,
                    frame.machine_state,
                    frame.bt_c,
                    frame.et_c,
                    frame.ror_c_per_min,
                    frame.gas_pct,
                    frame.drum_pressure_pa,
                    frame.drum_speed_pct,
                    estimated_state.Tb,
                    estimated_state.RoR * 60.0,
                    estimated_state.E_drum,
                    estimated_state.M,
                    estimated_state.P_int,
                    estimated_state.p_mai,
                    estimated_state.p_dev,
                    estimated_state.V_loss,
                    estimated_state.S_struct,
                    current_control.gas_pct,
                    current_control.drum_pressure_pa,
                    current_control.drum_speed_pct,
                    recommendation.recommended_gas_pct,
                    recommendation.recommended_drum_pressure_pa,
                    recommendation.recommended_drum_speed_pct,
                    recommendation.predicted_clarity,
                    recommendation.predicted_sweetness,
                    recommendation.predicted_body,
                    recommendation.predicted_bitterness,
                    int(mpc_success),
                    mpc_objective,
                    mpc_status,
                    recommendation.message,
                ]
            )