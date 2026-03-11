from __future__ import annotations

from pathlib import Path

from roastos.controller import RoastController
from roastos.state import initial_state
from roastos.types import Control

"""This script demonstrates how to use the RoastController to 
evaluate different control sequences and select the best one based on predicted flavor outcomes.    
It initializes the controller with the trained models, defines a target  flavor profile and 
session/coffee context, and then evaluates multiple candidate control sequences by simulating 
their trajectories, extracting features, making flavor predictions, and computing costs. 
Finally, it prints out the evaluation results for all candidates and highlights the best 
option selected by the controller. This serves as a practical example of how the controller 
can be used to guide roast decisions based on predicted flavor profiles."""

def build_candidate_control_sequences() -> list[list[Control]]:
    """
    Create a small set of sensible candidate future control plans.
    Each plan is a short sequence over the prediction horizon.
    """
    return [
        # Option 1: Hold stronger energy
        [Control(75, 60, 65)] * 8,

        # Option 2: Slight reduction in gas, mild airflow increase
        [Control(72, 63, 65)] * 8,

        # Option 3: More filter-clean style move
        [Control(70, 65, 65)] * 8,

        # Option 4: Reduce gas more and raise airflow more
        [Control(68, 68, 65)] * 8,

        # Option 5: Aggressive cleanup, risk of going too thin
        [Control(65, 72, 65)] * 8,
    ]


def pretty_print_option(idx: int, evaluation) -> None:
    first_control = evaluation.control_sequence[0]
    pf = evaluation.predicted_flavor
    rf = evaluation.roast_features

    print(f"\nOption {idx}")
    print("-" * 60)
    print(
        f"First control -> gas={first_control.gas:.1f}%, "
        f"airflow={first_control.airflow:.1f}%, "
        f"drum={first_control.drum_speed:.1f}%"
    )
    print(f"Cost: {evaluation.cost:.4f}")

    print("Predicted flavor:")
    print(
        f"  clarity={pf['clarity']:.3f}, "
        f"sweetness={pf['sweetness']:.3f}, "
        f"body={pf['body']:.3f}, "
        f"bitterness={pf['bitterness']:.3f}"
    )

    print("Predicted roast structure:")
    print(
        f"  dry={rf['dry']:.3f}, "
        f"maillard={rf['maillard']:.3f}, "
        f"dev={rf['dev']:.3f}, "
        f"ror_fc={rf['ror_fc']:.3f}"
    )
    print(
        f"  volatile_loss={rf['volatile_loss']:.3f}, "
        f"structure={rf['structure']:.3f}, "
        f"crash_index={rf['crash_index']:.3f}, "
        f"flick_index={rf['flick_index']:.3f}"
    )
    print(
        f"  time_to_yellow_s={rf['time_to_yellow_s']}, "
        f"time_to_fc_s={rf['time_to_fc_s']}, "
        f"dev_time_s={rf['dev_time_s']}, "
        f"delta_bt_fc_to_drop_c={rf['delta_bt_fc_to_drop_c']:.3f}"
    )


def main() -> None:
    print("\nRoastOS v1 Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Current roast state
    # ------------------------------------------------------------------
    state = initial_state()

    print("\nInitial roast state:")
    print(
        f"  Tb={state.Tb:.2f}, "
        f"E_drum={state.E_drum:.3f}, "
        f"p_dry={state.p_dry:.3f}, "
        f"p_mai={state.p_mai:.3f}, "
        f"p_dev={state.p_dev:.3f}, "
        f"V_loss={state.V_loss:.3f}, "
        f"S_struct={state.S_struct:.3f}"
    )

    # ------------------------------------------------------------------
    # 2. Desired flavor target
    # ------------------------------------------------------------------
    target_flavor = {
        "clarity": 0.90,
        "sweetness": 0.75,
        "body": 0.35,
        "bitterness": 0.15,
    }

    print("\nTarget flavor:")
    for k, v in target_flavor.items():
        print(f"  {k}: {v:.3f}")

    # ------------------------------------------------------------------
    # 3. Session context
    # These values must match the categories seen during training
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Coffee context
    # These values must also match training schema/categories
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Candidate control sequences
    # ------------------------------------------------------------------
    candidate_control_sequences = build_candidate_control_sequences()

    print(f"\nGenerated {len(candidate_control_sequences)} candidate control sequences.")

    # ------------------------------------------------------------------
    # 6. Controller
    # ------------------------------------------------------------------
    controller = RoastController(model_dir=Path("artifacts/models"))

    best, evaluations = controller.choose_best_option(
        initial_state=state,
        candidate_control_sequences=candidate_control_sequences,
        target_flavor=target_flavor,
        session_context=session_context,
        coffee_context=coffee_context,
    )

    # ------------------------------------------------------------------
    # 7. Print all options
    # ------------------------------------------------------------------
    print("\nCandidate evaluations:")
    print("=" * 60)

    for idx, evaluation in enumerate(evaluations, start=1):
        pretty_print_option(idx, evaluation)

    # ------------------------------------------------------------------
    # 8. Print best recommendation
    # ------------------------------------------------------------------
    best_control = best.control_sequence[0]

    print("\nBest option selected")
    print("=" * 60)
    print(
        f"Recommended next control -> "
        f"gas={best_control.gas:.1f}%, "
        f"airflow={best_control.airflow:.1f}%, "
        f"drum={best_control.drum_speed:.1f}%"
    )
    print(f"Expected cost: {best.cost:.4f}")
    print("Expected flavor:")
    for k, v in best.predicted_flavor.items():
        print(f"  {k}: {v:.3f}")

    print("\nExpected roast structure:")
    for k, v in best.roast_features.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()