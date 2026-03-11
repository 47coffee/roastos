from pathlib import Path

from roastos.controller import RoastController
from roastos.state import initial_state
from roastos.types import Control

"""This script demonstrates how to use the RoastController to 
evaluate different control sequences and select the best one based on predicted flavor outcomes. 
 It initializes the controller with the trained models, defines a target 
 flavor profile and session/coffee context, and then evaluates multiple candidate control 
 sequences by simulating their trajectories, extracting features, making flavor predictions, and computing costs. 
 Finally, it prints out the evaluation results for all candidates and highlights the best option selected by the controller. 
 This serves as a practical example of how the controller can be used to guide roast decisions based on predicted flavor profiles."""
def main() -> None:
    controller = RoastController(model_dir=Path("artifacts/models"))
    state = initial_state()

    target_flavor = {
        "clarity": 0.90,
        "sweetness": 0.75,
        "body": 0.35,
        "bitterness": 0.15,
    }

    session_context = {
        "machine_id": "PROBAT_P12_01",
        "coffee_id": "RW_SHG_WASHED_01",
        "operator_id": "simone",
        "style_profile": "filter_clarity",
        "brew_method": "cupping",
        "batch_size_kg": 6.0,
        "charge_temp_c": 205.0,
        "drop_temp_c": 204.5,
        "duration_s": 570,
        "ambient_temp_c": 21.0,
        "ambient_rh_pct": 48.0,
        "intent_clarity": 0.90,
        "intent_sweetness": 0.75,
        "intent_body": 0.35,
        "intent_bitterness": 0.15,
        "timestamp_start": "2026-03-10T09:00:00",
    }

    coffee_context = {
        "origin": "Rwanda",
        "process": "washed",
        "variety": "Bourbon",
        "density": 0.78,
        "moisture": 0.11,
        "water_activity": 0.54,
        "screen_size": 16.5,
        "altitude_m": 1850,
    }

    candidate_control_sequences = [
        # Option A: keep stronger energy
        [Control(75, 60, 65)] * 8,
        # Option B: reduce gas, raise airflow
        [Control(70, 65, 65)] * 8,
        # Option C: reduce more aggressively
        [Control(65, 70, 65)] * 8,
    ]

    best, evaluations = controller.choose_best_option(
        initial_state=state,
        candidate_control_sequences=candidate_control_sequences,
        target_flavor=target_flavor,
        session_context=session_context,
        coffee_context=coffee_context,
    )

    print("\nAll candidate evaluations:")
    for idx, ev in enumerate(evaluations, start=1):
        print(f"\nOption {idx}")
        print(f"  Cost: {ev.cost:.4f}")
        print(f"  Predicted flavor: {ev.predicted_flavor}")
        print(f"  Roast features: {ev.roast_features}")
        print(f"  First control: {ev.control_sequence[0]}")

    print("\nBest option selected:")
    print(f"  Cost: {best.cost:.4f}")
    print(f"  Predicted flavor: {best.predicted_flavor}")
    print(f"  Recommended next control: {best.control_sequence[0]}")


if __name__ == "__main__":
    main()