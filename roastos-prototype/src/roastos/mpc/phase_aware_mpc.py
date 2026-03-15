from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from roastos.config import load_settings
from roastos.mpc.control_grid import CandidateSequence, build_blocked_control_sequences
from roastos.mpc.target_profile import TargetTrajectory
from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.sim_types import RoastContext, RoastControl, RoastSimState
from roastos.simulator.state_estimator import RoastStateEstimator


FlavourPredictor = Callable[[list[RoastSimState], Optional[RoastContext]], dict[str, float]]


@dataclass(frozen=True)
class MPCRecommendation:
    control: RoastControl
    objective: float
    best_index: int


@dataclass(frozen=True)
class OnlineMPCStepResult:
    estimated_state: RoastSimState
    recommendation: MPCRecommendation


class PhaseAwareMPC:
    """
    V4.0 thermal MPC
    - Uses observer-estimated current state
    - Optimizes gas + pressure over a blocked horizon
    - Scores BT/ET tracking + move penalties

    V4.1
    - terminal drop objectives / context-aware plant

    V4.2
    - flavour-cost hooks
    """

    def __init__(
        self,
        simulator: CalibratedRoasterSimulator,
        flavour_predictor: FlavourPredictor | None = None,
    ):
        self.simulator = simulator
        self.settings = load_settings()
        self.flavour_predictor = flavour_predictor

    def _score_tracking_terms(
        self,
        states: list[RoastSimState],
        target: TargetTrajectory,
    ) -> float:
        cfg = self.settings.mpc
        target_points = target.slice(len(states) - 1)

        score = 0.0

        for i, target_pt in enumerate(target_points, start=1):
            s = states[i]
            bt_err = s.bt - target_pt.bt
            et_err = s.et - target_pt.et

            score += cfg.bt_track_weight * (bt_err ** 2)
            score += cfg.et_track_weight * (et_err ** 2)

        if target_points:
            terminal_state = states[len(target_points)]
            terminal_target = target_points[-1]

            score += cfg.terminal_bt_weight * ((terminal_state.bt - terminal_target.bt) ** 2)
            score += cfg.terminal_et_weight * ((terminal_state.et - terminal_target.et) ** 2)

        return float(score)

    def _score_move_penalties(
        self,
        candidate: CandidateSequence,
        current_control: RoastControl,
    ) -> float:
        cfg = self.settings.mpc
        first = candidate.controls[0]

        score = 0.0
        score += cfg.gas_move_penalty * ((first.gas - current_control.gas) ** 2)
        score += cfg.pressure_move_penalty * ((first.pressure - current_control.pressure) ** 2)
        return float(score)

    def _score_terminal_terms(
        self,
        states: list[RoastSimState],
        target: TargetTrajectory,
        context: Optional[RoastContext],
    ) -> float:
        cfg = self.settings.mpc
        score = 0.0

        if target.terminal is not None:
            terminal_outputs = self.simulator.predict_terminal_outputs(states, context=context)

            if target.terminal.drop_bt is not None:
                score += cfg.terminal_bt_weight * (
                    (terminal_outputs.drop_bt - target.terminal.drop_bt) ** 2
                )

            if target.terminal.drop_weight_kg is not None and terminal_outputs.drop_weight_kg is not None:
                score += cfg.terminal_drop_weight_weight * (
                    (terminal_outputs.drop_weight_kg - target.terminal.drop_weight_kg) ** 2
                )

        return float(score)

    def _score_flavour_terms(
        self,
        states: list[RoastSimState],
        target: TargetTrajectory,
        context: Optional[RoastContext],
    ) -> float:
        if self.flavour_predictor is None or not target.flavour_intent:
            return 0.0

        score = 0.0
        flavour_pred = self.flavour_predictor(states, context)
        flavour_weights = target.flavour_weights or {}

        for key, desired in target.flavour_intent.items():
            if key not in flavour_pred:
                continue
            w = float(flavour_weights.get(key, 1.0))
            score += w * ((float(flavour_pred[key]) - float(desired)) ** 2)

        return float(score)

    def _score_rollout(
        self,
        states: list[RoastSimState],
        target: TargetTrajectory,
        candidate: CandidateSequence,
        current_control: RoastControl,
        context: Optional[RoastContext],
    ) -> float:
        score = 0.0
        score += self._score_tracking_terms(states=states, target=target)
        score += self._score_move_penalties(candidate=candidate, current_control=current_control)
        score += self._score_terminal_terms(states=states, target=target, context=context)
        score += self._score_flavour_terms(states=states, target=target, context=context)
        return float(score)

    def recommend(
        self,
        current_state: RoastSimState,
        current_control: RoastControl,
        target: TargetTrajectory,
        phase_overrides: list[str] | None = None,
        context: RoastContext | None = None,
    ) -> MPCRecommendation:
        """
        Recommend the next control given the *current estimated state*.
        """
        candidates = build_blocked_control_sequences(
            current_gas=current_control.gas,
            current_pressure=current_control.pressure,
            drum_speed=current_control.drum_speed,
        )

        best_score = float("inf")
        best_idx = -1
        best_control = current_control

        for i, candidate in enumerate(candidates):
            rollout = self.simulator.rollout(
                initial_state=current_state,
                controls=candidate.controls,
                phase_overrides=phase_overrides,
                context=context,
            )

            score = self._score_rollout(
                states=rollout,
                target=target,
                candidate=candidate,
                current_control=current_control,
                context=context,
            )

            if score < best_score:
                best_score = score
                best_idx = i
                best_control = candidate.controls[0]

        return MPCRecommendation(
            control=best_control,
            objective=float(best_score),
            best_index=int(best_idx),
        )

    def observe_and_recommend(
        self,
        estimator: RoastStateEstimator,
        previous_state: RoastSimState,
        previous_control: RoastControl,
        measured_bt: float,
        measured_et: float,
        measured_ror: float,
        target: TargetTrajectory,
        context: RoastContext | None = None,
        phase_override: str | None = None,
    ) -> OnlineMPCStepResult:
        """
        V4.0 online control step:
        1. estimate current state from the latest measurements
        2. run MPC on that estimated state
        """
        estimated_state = estimator.estimate_next(
            state=previous_state,
            control=previous_control,
            measured_bt_next=measured_bt,
            measured_et_next=measured_et,
            measured_ror_next=measured_ror,
            context=context,
            phase_override=phase_override,
        )[1]

        recommendation = self.recommend(
            current_state=estimated_state,
            current_control=previous_control,
            target=target,
            phase_overrides=None,
            context=context,
        )

        return OnlineMPCStepResult(
            estimated_state=estimated_state,
            recommendation=recommendation,
        )

