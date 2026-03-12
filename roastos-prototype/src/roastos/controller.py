from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roastos.dynamics import step_dynamics
from roastos.features import extract_features
from roastos.inference_row_builder import build_inference_row
from roastos.objective import flavor_cost
from roastos.predictor import FlavorPredictor
from roastos.types import Control, RoastState

"""This module defines the RoastController class, which is responsible for simulating future trajectories of the 
roasting process under different control sequences, extracting features from those trajectories, 
building inference rows for flavor prediction, and ultimately evaluating and selecting the best 
control sequence based on predicted flavor outcomes and associated costs. 
The controller uses a trained flavor predictor model to make informed decisions about 
roast adjustments in order to achieve desired flavor profiles."""         

@dataclass
class ControlEvaluation:
    control_sequence: list[Control]
    trajectory_states: list[RoastState]
    final_state: RoastState
    roast_features: dict[str, Any]
    inference_row: dict[str, Any]
    predicted_flavor: dict[str, float]
    cost: float


class RoastController:
    """
    RoastOS v1 controller:
    - simulate future control options
    - extract roast structure
    - build ML-ready inference row
    - predict flavor with trained models
    - compute cost
    - choose best action sequence
    """

    def __init__(self, model_dir: str | Path):
        self.predictor = FlavorPredictor(model_dir=model_dir)

    def simulate_trajectory(
        self,
        initial_state: RoastState,
        control_sequence: list[Control],
        coffee_context: dict[str, Any] | None = None,
    ) -> list[RoastState]:
        states = [initial_state]
        current_state = initial_state

        for control in control_sequence:
            current_state = step_dynamics(
                current_state,
                control,
                coffee_context=coffee_context,
            )
            states.append(current_state)

        return states

    def evaluate_option(
        self,
        *,
        initial_state: RoastState,
        control_sequence: list[Control],
        target_flavor: dict[str, float],
        session_context: dict[str, Any],
        coffee_context: dict[str, Any],
    ) -> ControlEvaluation:
        states = self.simulate_trajectory(
            initial_state=initial_state,
            control_sequence=control_sequence,
            coffee_context=coffee_context,
        )

        roast_features = extract_features(states)

        inference_row = build_inference_row(
            roast_features=roast_features,
            session_context=session_context,
            coffee_context=coffee_context,
            required_feature_names=self.predictor.get_required_features(),
        )

        predicted_flavor = self.predictor.predict_row(inference_row).to_dict()
        cost = flavor_cost(predicted_flavor, target_flavor)

        return ControlEvaluation(
            control_sequence=control_sequence,
            trajectory_states=states,
            final_state=states[-1],
            roast_features=roast_features,
            inference_row=inference_row,
            predicted_flavor=predicted_flavor,
            cost=cost,
        )

    def choose_best_option(
        self,
        *,
        initial_state: RoastState,
        candidate_control_sequences: list[list[Control]],
        target_flavor: dict[str, float],
        session_context: dict[str, Any],
        coffee_context: dict[str, Any],
    ) -> tuple[ControlEvaluation, list[ControlEvaluation]]:
        evaluations: list[ControlEvaluation] = []

        for sequence in candidate_control_sequences:
            evaluation = self.evaluate_option(
                initial_state=initial_state,
                control_sequence=sequence,
                target_flavor=target_flavor,
                session_context=session_context,
                coffee_context=coffee_context,
            )
            evaluations.append(evaluation)

        best = min(evaluations, key=lambda e: e.cost)
        return best, evaluations