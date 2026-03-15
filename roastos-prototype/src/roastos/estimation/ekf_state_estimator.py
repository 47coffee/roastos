from __future__ import annotations

from dataclasses import dataclass

from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.sim_types import RoastContext, RoastControl, RoastSimState


@dataclass
class EKFSettings:
    """
    Lightweight EKF-style observer gains.

    This is intentionally simple:
    - predict with the simulator
    - correct observed BT/ET/RoR with diagonal gains
    - keep latent states from model prediction

    Good enough for V4.0 scaffolding.
    """
    k_bt: float = 0.35
    k_et: float = 0.25
    k_ror: float = 0.20


class RoastEKFStateEstimator:
    def __init__(self, simulator: CalibratedRoasterSimulator, settings: EKFSettings | None = None):
        self.simulator = simulator
        self.settings = settings or EKFSettings()

    def initialize(
        self,
        t_sec: float,
        measured_bt: float,
        measured_et: float,
        measured_ror: float,
        control: RoastControl,
        context: RoastContext | None = None,
        e_drum_raw: float = 0.0,
        phase: str | None = None,
    ) -> RoastSimState:
        return self.simulator.build_initial_state(
            t_sec=t_sec,
            bt=measured_bt,
            et=measured_et,
            ror=measured_ror,
            gas=control.gas,
            pressure=control.pressure,
            drum_speed=control.drum_speed,
            e_drum_raw=e_drum_raw,
            context=context,
            phase=phase,
        )

    def predict(
        self,
        previous_state: RoastSimState,
        previous_control: RoastControl,
        context: RoastContext | None = None,
        phase_override: str | None = None,
    ) -> RoastSimState:
        return self.simulator.step(
            previous_state,
            previous_control,
            context=context,
            phase_override=phase_override,
        ).next_state

    def update(
        self,
        predicted_state: RoastSimState,
        measured_bt: float,
        measured_et: float,
        measured_ror: float | None = None,
    ) -> RoastSimState:
        k_bt = self.settings.k_bt
        k_et = self.settings.k_et
        k_ror = self.settings.k_ror

        bt = predicted_state.bt + k_bt * (float(measured_bt) - predicted_state.bt)
        et = predicted_state.et + k_et * (float(measured_et) - predicted_state.et)

        if measured_ror is None:
            ror = predicted_state.ror
        else:
            ror = predicted_state.ror + k_ror * (float(measured_ror) - predicted_state.ror)

        corrected = predicted_state.copy()
        corrected.bt = bt
        corrected.et = et
        corrected.ror = ror
        corrected.phase = self.simulator._infer_phase_open_loop(corrected)
        return corrected

    def estimate(
        self,
        previous_state: RoastSimState,
        previous_control: RoastControl,
        measured_bt: float,
        measured_et: float,
        measured_ror: float | None = None,
        context: RoastContext | None = None,
        phase_override: str | None = None,
    ) -> RoastSimState:
        predicted = self.predict(
            previous_state=previous_state,
            previous_control=previous_control,
            context=context,
            phase_override=phase_override,
        )
        return self.update(
            predicted_state=predicted,
            measured_bt=measured_bt,
            measured_et=measured_et,
            measured_ror=measured_ror,
        )
