from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional

from .phase_logic import infer_phase_from_bt
from .sim_types import (
    RoastContext,
    RoastControl,
    RoastSimState,
    SimStepResult,
    SimulatorParams,
    TerminalOutputs,
)


class CalibratedRoasterSimulator:
    """
    V3/V4 RoastOS forward simulator.

    Design rule:
    - keep frozen V3.0 BT/ET phase equations intact
    - add V4 auxiliary states and V4.1 context hooks around them
    - do not silently reopen the calibrated plant
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
        return infer_phase_from_bt(state.bt)

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

    def _clip01(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _ratio(self, value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return self._clip01((float(value) - lo) / (hi - lo))

    def _effective_bean_start_temp(self, context: Optional[RoastContext]) -> Optional[float]:
        if context is None:
            return None
        if context.bean_start_temp_c is not None:
            return float(context.bean_start_temp_c)
        if context.start_temp_c is not None:
            return float(context.start_temp_c)
        return None

    def _effective_charge_temp(self, context: Optional[RoastContext]) -> Optional[float]:
        if context is None:
            return None
        if context.charge_temp_c is not None:
            return float(context.charge_temp_c)
        return None

    def initial_moisture_burden(self, context: Optional[RoastContext]) -> float:
        """
        Moisture burden should respond primarily to:
        - bean mass
        - bean start temperature

        It should NOT directly use machine charge temperature. That belongs more to
        machine-side initial condition / latent energy logic.
        """
        base = 1.0
        if context is None:
            return base

        if context.start_weight_kg is not None and context.start_weight_kg > 0:
            base *= float(context.start_weight_kg) / float(self.params.reference_charge_weight_kg)

        bean_start_temp = self._effective_bean_start_temp(context)
        if bean_start_temp is not None:
            temp_gap = float(self.params.reference_bean_start_temp_c) - float(bean_start_temp)
            base *= 1.0 + max(temp_gap, -10.0) / 100.0

        return max(0.25, min(2.0, base))

    def initial_progress_state(self, bt: float) -> tuple[float, float, float]:
        p_dry = self._ratio(bt, self.params.progress_drying_bt_start, self.params.progress_drying_bt_end)
        p_mai = self._ratio(bt, self.params.progress_maillard_bt_start, self.params.progress_maillard_bt_end)
        p_dev = self._ratio(bt, self.params.progress_development_bt_start, self.params.progress_development_bt_end)
        return p_dry, p_mai, p_dev

    def _adjust_initial_e_drum_raw_for_charge_temp(
        self,
        e_drum_raw: float,
        context: Optional[RoastContext],
    ) -> float:
        """
        Optional initial adjustment of machine-side thermal state using charge temp.

        This is intentionally mild and only active when context dynamics are enabled.
        It does not reopen the frozen V3.0 phase equations; it only shifts the
        initial latent machine condition.
        """
        if not self.params.enable_context_dynamics or context is None:
            return float(e_drum_raw)

        charge_temp = self._effective_charge_temp(context)
        if charge_temp is None:
            return float(e_drum_raw)

        temp_gap = float(charge_temp) - float(self.params.reference_charge_temp_c)
        # small scaling only; keep conservative
        return float(e_drum_raw) + 0.02 * temp_gap

    def build_initial_state(
        self,
        t_sec: float,
        bt: float,
        et: float,
        ror: float,
        gas: float,
        pressure: float,
        drum_speed: float,
        e_drum_raw: float,
        context: Optional[RoastContext] = None,
        phase: Optional[str] = None,
    ) -> RoastSimState:
        e_drum_raw_adj = self._adjust_initial_e_drum_raw_for_charge_temp(e_drum_raw, context)
        e_drum = self._standardize_e_drum(e_drum_raw_adj)
        p_dry, p_mai, p_dev = self.initial_progress_state(bt)

        return RoastSimState(
            t_sec=float(t_sec),
            bt=float(bt),
            et=float(et),
            ror=float(ror),
            e_drum_raw=float(e_drum_raw_adj),
            e_drum=float(e_drum),
            m_burden=self.initial_moisture_burden(context),
            p_dry=p_dry,
            p_mai=p_mai,
            p_dev=p_dev,
            phase=phase if phase is not None else infer_phase_from_bt(bt),
            gas=float(gas),
            pressure=float(pressure),
            drum_speed=float(drum_speed),
            bt_prev=float(bt),
            et_prev=float(et),
            prev_pressure=float(pressure),
        )

    def _context_response_factor(self, context: Optional[RoastContext]) -> float:
        if not self.params.enable_context_dynamics or context is None:
            return 1.0

        factor = 1.0

        if context.start_weight_kg is not None and context.start_weight_kg > 0:
            mass_factor = float(self.params.reference_charge_weight_kg) / float(context.start_weight_kg)
            factor *= mass_factor

        factor = max(self.params.min_context_response_scale, min(self.params.max_context_response_scale, factor))
        return factor

    def _predict_et_step(
        self,
        state: RoastSimState,
        phase: str,
        gas_used: float,
        pressure_used: float,
        ror_used: float,
        e_drum_used: float,
        et_used: float,
        context: Optional[RoastContext] = None,
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

        if self.params.moisture_et_drag_coeff != 0.0:
            et_step -= float(self.params.moisture_et_drag_coeff) * float(state.m_burden)

        et_step *= self._context_response_factor(context)
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
        context: Optional[RoastContext] = None,
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

        if self.params.moisture_bt_drag_coeff != 0.0:
            bt_delta -= float(self.params.moisture_bt_drag_coeff) * float(state.m_burden)

        bt_delta *= self._context_response_factor(context)
        return bt_delta

    def _update_ror(self, prev_ror: float, bt_now: float, bt_next: float) -> float:
        ror_raw = 60.0 * (float(bt_next) - float(bt_now)) / float(self.params.dt_sec)
        a = float(self.params.ror_filter_alpha)
        return a * float(prev_ror) + (1.0 - a) * ror_raw

    def _update_moisture_burden(
        self,
        state: RoastSimState,
        bt_next: float,
        et_next: float,
        context: Optional[RoastContext] = None,
    ) -> float:
        heat_drive = max((float(bt_next) + float(et_next)) / 2.0 - 100.0, 0.0)
        decay = self.params.moisture_decay_rate + self.params.moisture_heat_coeff * (heat_drive / 100.0)

        if context is not None and context.start_weight_kg is not None and context.start_weight_kg > 0:
            decay *= min(1.5, max(0.6, self.params.reference_charge_weight_kg / float(context.start_weight_kg)))

        next_m = float(state.m_burden) - decay
        return max(0.0, next_m)

    def _update_progress_states(
        self,
        state: RoastSimState,
        bt_next: float,
    ) -> tuple[float, float, float]:
        p_dry = max(
            state.p_dry,
            self._ratio(bt_next, self.params.progress_drying_bt_start, self.params.progress_drying_bt_end),
        )
        p_mai = max(
            state.p_mai,
            self._ratio(bt_next, self.params.progress_maillard_bt_start, self.params.progress_maillard_bt_end),
        )
        p_dev = max(
            state.p_dev,
            self._ratio(bt_next, self.params.progress_development_bt_start, self.params.progress_development_bt_end),
        )
        return p_dry, p_mai, p_dev

    def step(
        self,
        state: RoastSimState,
        control: RoastControl,
        teacher_forced_et: Optional[float] = None,
        phase_override: Optional[str] = None,
        teacher_forced_ror: Optional[float] = None,
        context: Optional[RoastContext] = None,
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
                context=context,
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
            context=context,
        )
        bt_next = float(state.bt) + float(bt_delta)

        ror_next = self._update_ror(state.ror, state.bt, bt_next)
        m_next = self._update_moisture_burden(state, bt_next, et_next, context=context)
        p_dry_next, p_mai_next, p_dev_next = self._update_progress_states(state, bt_next)

        next_state = RoastSimState(
            t_sec=float(state.t_sec) + float(self.params.dt_sec),
            bt=bt_next,
            et=et_next,
            ror=ror_next,
            e_drum_raw=e_drum_raw_next,
            e_drum=e_drum_next,
            m_burden=m_next,
            p_dry=p_dry_next,
            p_mai=p_mai_next,
            p_dev=p_dev_next,
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
        context: Optional[RoastContext] = None,
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
                context=context,
            )
            current = result.next_state
            states.append(current)

        return states

    def predict_terminal_outputs(
        self,
        states: List[RoastSimState],
        context: Optional[RoastContext] = None,
    ) -> TerminalOutputs:
        last = states[-1]
        drop_weight_kg = None
        loss_fraction = None

        if context is not None and context.start_weight_kg is not None:
            loss_fraction = (
                self.params.drop_weight_moisture_coeff * max(0.0, 1.0 - last.m_burden)
                + self.params.drop_weight_maillard_coeff * last.p_mai
                + self.params.drop_weight_development_coeff * last.p_dev
            )
            loss_fraction = max(0.0, min(0.30, float(loss_fraction)))
            drop_weight_kg = max(0.0, float(context.start_weight_kg) * (1.0 - loss_fraction))

        return TerminalOutputs(
            drop_bt=float(last.bt),
            drop_time_s=float(last.t_sec),
            drop_weight_kg=drop_weight_kg,
            loss_fraction=loss_fraction,
            dtr=last.p_dev,
        )

    def state_to_dict(self, state: RoastSimState) -> dict:
        return asdict(state)

