from __future__ import annotations

from dataclasses import dataclass

from roastos.config import load_settings
from .calibrated_simulator import CalibratedRoasterSimulator
from .sim_types import RoastContext, RoastControl, RoastSimState


@dataclass
class EstimatorGains:
    k_bt: float = 0.35
    k_et: float = 0.25
    k_ror: float = 0.20
    k_e_drum: float = 0.00
    k_m_burden: float = 0.00


class RoastStateEstimator:
    """
    Lightweight RoastOS observer for V3.2.

    Design:
    - predict next state with the frozen simulator plant
    - correct observed channels BT / ET / RoR with simple gains
    - keep latent/process states mostly model-driven

    This is EKF-style architecture without a full covariance implementation yet.
    It is intentionally simple and stable as a pre-MPC observer layer.
    """

    def __init__(
        self,
        simulator: CalibratedRoasterSimulator,
        gains: EstimatorGains | None = None,
    ):
        self.simulator = simulator
        self.settings = load_settings()

        if gains is None:
            gains = EstimatorGains(
                k_bt=float(self.settings.ekf.k_bt),
                k_et=float(self.settings.ekf.k_et),
                k_ror=float(self.settings.ekf.k_ror),
                k_e_drum=0.00,
                k_m_burden=0.00,
            )
        self.gains = gains

    def _default_bean_start_temp_c(self) -> float:
        value = getattr(self.settings.context_model, "default_bean_start_temp_c", None)
        if value is not None:
            return float(value)
        legacy = getattr(self.settings.context_model, "default_start_temp_c", None)
        if legacy is not None:
            return float(legacy)
        return 25.0

    def _default_charge_temp_c(self) -> float:
        value = getattr(self.settings.context_model, "default_charge_temp_c", None)
        if value is not None:
            return float(value)
        return 230.0

    def infer_initial_bt(
        self,
        measured_bt: float,
        context: RoastContext | None,
    ) -> float:
        """
        At charge, measured BT can be a hot transient probe/chamber effect rather than
        true bean bulk temperature. Prefer bean start temp if available.
        """
        if context is not None and context.bean_start_temp_c is not None:
            return float(context.bean_start_temp_c)

        if context is not None and context.start_temp_c is not None:
            return float(context.start_temp_c)

        return self._default_bean_start_temp_c()

    def infer_initial_et(
        self,
        measured_et: float,
        context: RoastContext | None,
    ) -> float:
        if context is not None and context.charge_temp_c is not None:
            return float(context.charge_temp_c)
        return float(measured_et) if measured_et is not None else self._default_charge_temp_c()

    def initialize(
        self,
        t_sec: float,
        measured_bt: float,
        measured_et: float,
        measured_ror: float,
        control: RoastControl,
        e_drum_raw: float,
        context: RoastContext | None = None,
        phase: str | None = None,
    ) -> RoastSimState:
        bt0 = self.infer_initial_bt(measured_bt, context)
        et0 = self.infer_initial_et(measured_et, context)
        ror0 = 0.0

        state = self.simulator.build_initial_state(
            t_sec=float(t_sec),
            bt=float(bt0),
            et=float(et0),
            ror=float(ror0),
            gas=float(control.gas),
            pressure=float(control.pressure),
            drum_speed=float(control.drum_speed),
            e_drum_raw=float(e_drum_raw),
            context=context,
            phase=phase,
        )
        return state

    def predict(
        self,
        state: RoastSimState,
        control: RoastControl,
        context: RoastContext | None = None,
        teacher_forced_et: float | None = None,
        teacher_forced_ror: float | None = None,
        phase_override: str | None = None,
    ) -> RoastSimState:
        result = self.simulator.step(
            state=state,
            control=control,
            teacher_forced_et=teacher_forced_et,
            teacher_forced_ror=teacher_forced_ror,
            phase_override=phase_override,
            context=context,
        )
        return result.next_state

    def update(
        self,
        predicted_state: RoastSimState,
        measured_bt: float,
        measured_et: float,
        measured_ror: float,
    ) -> RoastSimState:
        corrected = predicted_state.copy()

        corrected.bt = corrected.bt + self.gains.k_bt * (float(measured_bt) - corrected.bt)
        corrected.et = corrected.et + self.gains.k_et * (float(measured_et) - corrected.et)
        corrected.ror = corrected.ror + self.gains.k_ror * (float(measured_ror) - corrected.ror)

        corrected.phase = self.simulator._infer_phase_open_loop(corrected)
        corrected.bt_prev = predicted_state.bt_prev
        corrected.et_prev = predicted_state.et_prev
        corrected.prev_pressure = predicted_state.prev_pressure

        return corrected

    def estimate_next(
        self,
        state: RoastSimState,
        control: RoastControl,
        measured_bt_next: float,
        measured_et_next: float,
        measured_ror_next: float,
        context: RoastContext | None = None,
        teacher_forced_et: float | None = None,
        teacher_forced_ror: float | None = None,
        phase_override: str | None = None,
    ) -> tuple[RoastSimState, RoastSimState]:
        """
        Returns:
        - predicted_state (before correction)  -> use this for replay error metrics
        - corrected_state (after measurement update) -> use this for next step
        """
        predicted_state = self.predict(
            state=state,
            control=control,
            context=context,
            teacher_forced_et=teacher_forced_et,
            teacher_forced_ror=teacher_forced_ror,
            phase_override=phase_override,
        )
        corrected_state = self.update(
            predicted_state=predicted_state,
            measured_bt=measured_bt_next,
            measured_et=measured_et_next,
            measured_ror=measured_ror_next,
        )
        return predicted_state, corrected_state

