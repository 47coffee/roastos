from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional

from .sim_types import (
    RoastControl,
    RoastSimState,
    SimStepResult,
    SimulatorParams,
)


class CalibratedRoasterSimulator:
    """
    Calibration-aligned RoastOS forward simulator.

    Runtime order per step:
    1. normalize inputs
    2. update latent e_drum state
    3. predict ET step (unless ET is teacher-forced)
    4. predict BT delta
    5. update RoR from BT movement

    This version consumes the richer ET v2 phase models, including lagged gap,
    lagged pressure, positive pressure step, and ET level decay. :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, params: SimulatorParams):
        self.params = params

    def _normalize_gas(self, gas: float) -> float:
        gas_val = float(gas)
        if self.params.gas_already_normalized:
            return gas_val
        return gas_val / 100.0

    def _prepare_pressure(self, pressure: float) -> float:
        return float(pressure)

    def _compute_e_drum_raw(self, prev_raw: float, gas: float, pressure: float) -> float:
        return (
            self.params.latent_decay * float(prev_raw)
            + float(gas)
            - float(pressure) / self.params.latent_pressure_scale
        )

    def _standardize_e_drum(self, e_drum_raw: float) -> float:
        std = float(self.params.latent_std)
        if abs(std) < 1e-12:
            std = 1.0
        return (float(e_drum_raw) - float(self.params.latent_mean)) / std

    def _infer_phase_open_loop(self, state: RoastSimState) -> str:
        if state.bt < 160.0:
            return "drying"
        if state.bt < 196.0:
            return "maillard"
        return "development"

    def _clip_ror_for_model(self, ror: float) -> float:
        clip = float(self.params.ror_model_clip)
        return max(min(float(ror), clip), -clip)

    def _compute_bt_c_norm(self, bt: float) -> float:
        denom = float(self.params.bt_norm_denominator)
        if abs(denom) < 1e-12:
            denom = 1.0
        return (float(bt) - float(self.params.bt_norm_offset)) / denom

    def _compute_et_c_norm(self, et: float) -> float:
        denom = float(self.params.et_norm_denominator)
        if abs(denom) < 1e-12:
            denom = 1.0
        return float(et) / denom

    def _predict_et_step(
        self,
        state: RoastSimState,
        phase: str,
        gas_used: float,
        pressure_used: float,
        ror_used: float,
        e_drum_used: float,
        et_used: float,
    ) -> float:
        if phase not in self.params.phase_models:
            raise KeyError(f"Unknown phase '{phase}'. Available: {list(self.params.phase_models.keys())}")

        p = self.params.phase_models[phase].et

        et_gap = float(et_used) - float(state.bt)

        if state.et_prev is not None and state.bt_prev is not None:
            et_gap_lag1 = float(state.et_prev) - float(state.bt_prev)
        else:
            et_gap_lag1 = et_gap

        prev_pressure = float(state.prev_pressure) if state.prev_pressure is not None else float(state.pressure)
        pressure_delta_pos = max(float(pressure_used) - prev_pressure, 0.0)

        ror_for_model = self._clip_ror_for_model(ror_used)
        et_level = self._compute_et_c_norm(et_used)

        et_step = float(p.intercept)
        et_step += float(p.c_e_drum) * float(e_drum_used)
        et_step += float(p.c_gas) * float(gas_used)
        et_step += float(p.c_et_gap) * float(et_gap)
        et_step += float(p.c_et_gap_lag1) * float(et_gap_lag1)
        et_step += float(p.c_pressure) * float(pressure_used)
        et_step += float(p.c_pressure_lag1) * float(prev_pressure)
        et_step += float(p.c_pressure_delta_pos) * float(pressure_delta_pos)
        et_step += float(p.c_ror) * float(ror_for_model)
        et_step += float(p.c_et_level) * float(et_level)

        return et_step

    def _predict_bt_delta(
        self,
        state: RoastSimState,
        phase: str,
        et_used: float,
        gas_used: float,
        pressure_used: float,
        ror_used: float,
        e_drum_used: float,
    ) -> float:
        if phase not in self.params.phase_models:
            raise KeyError(f"Unknown phase '{phase}'. Available: {list(self.params.phase_models.keys())}")

        p = self.params.phase_models[phase].bt

        bt_c_norm = self._compute_bt_c_norm(state.bt)
        et_delta = float(et_used) - float(state.bt)
        ror_for_model = self._clip_ror_for_model(ror_used)

        bt_delta = float(p.intercept)
        bt_delta += float(p.c_e_drum) * float(e_drum_used)
        bt_delta += float(p.c_et_delta) * float(et_delta)
        bt_delta += float(p.c_bt_level) * float(bt_c_norm)
        bt_delta += float(p.c_ror) * float(ror_for_model)
        bt_delta += float(p.c_pressure_direct) * float(pressure_used)

        if self.params.include_gas_feature:
            bt_delta += float(p.c_gas) * float(gas_used)

        return bt_delta

    def _update_ror(self, prev_ror: float, bt_now: float, bt_next: float) -> float:
        ror_raw = 60.0 * (float(bt_next) - float(bt_now)) / float(self.params.dt_sec)
        a = float(self.params.ror_filter_alpha)
        return a * float(prev_ror) + (1.0 - a) * ror_raw

    def step(
        self,
        state: RoastSimState,
        control: RoastControl,
        teacher_forced_et: Optional[float] = None,
        phase_override: Optional[str] = None,
        teacher_forced_ror: Optional[float] = None,
    ) -> SimStepResult:
        phase = phase_override if phase_override is not None else self._infer_phase_open_loop(state)

        gas_used = self._normalize_gas(control.gas)
        pressure_used = self._prepare_pressure(control.pressure)

        e_drum_raw_next = self._compute_e_drum_raw(
            prev_raw=state.e_drum_raw,
            gas=gas_used,
            pressure=pressure_used,
        )
        e_drum_next = self._standardize_e_drum(e_drum_raw_next)

        et_used = float(teacher_forced_et) if teacher_forced_et is not None else float(state.et)
        ror_used = float(teacher_forced_ror) if teacher_forced_ror is not None else float(state.ror)

        if teacher_forced_et is not None:
            et_next = et_used
        else:
            et_step = self._predict_et_step(
                state=state,
                phase=phase,
                gas_used=gas_used,
                pressure_used=pressure_used,
                ror_used=ror_used,
                e_drum_used=e_drum_next,
                et_used=et_used,
            )
            et_next = float(state.et) + float(et_step)

        bt_delta = self._predict_bt_delta(
            state=state,
            phase=phase,
            et_used=et_used,
            gas_used=gas_used,
            pressure_used=pressure_used,
            ror_used=ror_used,
            e_drum_used=e_drum_next,
        )
        bt_next = float(state.bt) + float(bt_delta)

        ror_next = self._update_ror(state.ror, state.bt, bt_next)

        next_state = RoastSimState(
            t_sec=float(state.t_sec) + float(self.params.dt_sec),
            bt=bt_next,
            et=et_next,
            ror=ror_next,
            e_drum_raw=e_drum_raw_next,
            e_drum=e_drum_next,
            phase=phase,
            gas=gas_used,
            pressure=pressure_used,
            drum_speed=float(control.drum_speed),
            bt_prev=float(state.bt),
            et_prev=float(state.et),
            prev_pressure=float(state.pressure),
        )

        return SimStepResult(
            prev_state=state.copy(),
            control=control,
            next_state=next_state,
        )

    def rollout(
        self,
        initial_state: RoastSimState,
        controls: Iterable[RoastControl],
        teacher_forced_ets: Optional[Iterable[float]] = None,
        phase_overrides: Optional[Iterable[str]] = None,
        teacher_forced_rors: Optional[Iterable[float]] = None,
    ) -> List[RoastSimState]:
        states: List[RoastSimState] = [initial_state.copy()]
        current = initial_state.copy()

        controls_list = list(controls)
        ets_list = list(teacher_forced_ets) if teacher_forced_ets is not None else [None] * len(controls_list)
        phase_list = list(phase_overrides) if phase_overrides is not None else [None] * len(controls_list)
        ror_list = list(teacher_forced_rors) if teacher_forced_rors is not None else [None] * len(controls_list)

        for control, forced_et, phase_override, forced_ror in zip(
            controls_list, ets_list, phase_list, ror_list
        ):
            result = self.step(
                current,
                control,
                teacher_forced_et=forced_et,
                phase_override=phase_override,
                teacher_forced_ror=forced_ror,
            )
            current = result.next_state
            states.append(current)

        return states

    def state_to_dict(self, state: RoastSimState) -> dict:
        return asdict(state)