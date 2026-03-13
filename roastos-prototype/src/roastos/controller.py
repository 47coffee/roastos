from __future__ import annotations

from dataclasses import dataclass

from roastos.dynamics import step_dynamics
from roastos.flavor_model import predict_flavor
from roastos.types import RoastState, Control


@dataclass
class CandidateEvaluation:
    final_state: RoastState
    predicted_flavor: dict[str, float]
    flavor_cost: float
    structure_summary: dict[str, float]


class RoastController:
    """
    High-level trajectory evaluator for RoastOS.

    Role:
    - simulate candidate control sequences
    - predict final flavour
    - score flavour error vs target flavour
    - choose best candidate
    """

    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def _simulate_sequence(
        self,
        initial_state: RoastState,
        control_sequence: list[Control],
        coffee_context: dict,
        dt_s: float = 2.0,
    ) -> RoastState:
        state = initial_state

        for control in control_sequence:
            state = step_dynamics(
                state=state,
                control=control,
                coffee_context=coffee_context,
                dt_s=dt_s,
            )

        return state

    def _structure_summary(self, state: RoastState, moisture0: float = 0.11) -> dict[str, float]:
        pct_dry = max(0.0, min(1.0, 1.0 - state.M / moisture0))

        return {
            "dry": pct_dry,
            "maillard": state.p_mai,
            "dev": state.p_dev,
            "ror_fc": state.RoR * 60.0,
            "volatile_loss": state.V_loss,
            "structure": state.S_struct,
        }

    def _flavor_cost(
        self,
        predicted_flavor: dict[str, float],
        target_flavor: dict[str, float],
    ) -> float:
        # weighted squared error
        weights = {
            "clarity": 1.5,
            "sweetness": 1.3,
            "body": 1.0,
            "bitterness": 1.4,
            "acidity_quality": 1.0,
        }

        cost = 0.0

        for key, target_val in target_flavor.items():
            pred_val = predicted_flavor.get(key, 0.0)
            w = weights.get(key, 1.0)
            cost += w * (pred_val - target_val) ** 2

        return cost

    def evaluate_candidate(
        self,
        *,
        initial_state: RoastState,
        control_sequence: list[Control],
        target_flavor: dict[str, float],
        coffee_context: dict,
        dt_s: float = 2.0,
    ) -> CandidateEvaluation:
        final_state = self._simulate_sequence(
            initial_state=initial_state,
            control_sequence=control_sequence,
            coffee_context=coffee_context,
            dt_s=dt_s,
        )

        predicted_flavor = predict_flavor(final_state)
        structure_summary = self._structure_summary(
            final_state,
            moisture0=coffee_context.get("moisture", 0.11),
        )
        flavor_cost = self._flavor_cost(predicted_flavor, target_flavor)

        return CandidateEvaluation(
            final_state=final_state,
            predicted_flavor=predicted_flavor,
            flavor_cost=flavor_cost,
            structure_summary=structure_summary,
        )

    def choose_best_option(
        self,
        *,
        initial_state: RoastState,
        candidate_control_sequences: list[list[Control]],
        target_flavor: dict[str, float],
        session_context: dict,
        coffee_context: dict,
        dt_s: float = 2.0,
    ) -> tuple[CandidateEvaluation, int]:
        evaluations: list[CandidateEvaluation] = []

        for seq in candidate_control_sequences:
            evaluation = self.evaluate_candidate(
                initial_state=initial_state,
                control_sequence=seq,
                target_flavor=target_flavor,
                coffee_context=coffee_context,
                dt_s=dt_s,
            )
            evaluations.append(evaluation)

        best_idx = min(range(len(evaluations)), key=lambda i: evaluations[i].flavor_cost)

        return evaluations[best_idx], best_idx