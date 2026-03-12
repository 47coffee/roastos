from __future__ import annotations

from pathlib import Path

from roastos.controller import RoastController
from roastos.plotting import plot_candidate_trajectories
from roastos.state import initial_state
from roastos.types import Control


def build_candidate_control_sequences() -> list[list[Control]]:
    """
    Create a more separated set of candidate future control plans
    so the simulated trajectories diverge more clearly.
    """
    return [
        # Option 1: stronger energy
        [Control(80, 55, 65)] * 10,

        # Option 2: moderate energy
        [Control(75, 60, 65)] * 10,

        # Option 3: balanced filter move
        [Control(70, 65, 65)] * 10,

        # Option 4: cleaner / higher airflow
        [Control(65, 72, 65)] * 10,

        # Option 5: aggressive cleanup
        [Control(60, 80, 65)] * 10,
    ]


def pretty_print_option(idx: int, evaluation) -> None:
    first_control = evaluation.control_sequence[0]
    pf = evaluation.predicted_flavor
    rf = evaluation.roast_features

    def g(key, default=0.0):
        return rf.get(key, default)

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
        f"  dry={g('dry'):.3f}, "
        f"maillard={g('maillard'):.3f}, "
        f"dev={g('dev'):.3f}, "
        f"ror_fc={g('ror_fc'):.3f}"
    )
    print(
        f"  volatile_loss={g('volatile_loss'):.3f}, "
        f"structure={g('structure'):.3f}, "
        f"crash_index={g('crash_index'):.3f}, "
        f"flick_index={g('flick_index'):.3f}"
    )
    print(
        f"  time_to_yellow_s={g('time_to_yellow_s', 0)}, "
        f"time_to_fc_s={g('time_to_fc_s', 0)}, "
        f"dev_time_s={g('dev_time_s', 0)}, "
        f"delta_bt_fc_to_drop_c={g('delta_bt_fc_to_drop_c'):.3f}"
    )


def main() -> None:
    print("\nRoastOS v1 Demo")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # 1. Current roast state
    # ------------------------------------------------------------------
    state = initial_state()

    print("\nInitial roast state:")
    print(
        f"  Tb={state.Tb:.2f}, E_drum={state.E_drum:.3f}, "
        f"p_dry={state.p_dry:.3f}, p_mai={state.p_mai:.3f}, "
        f"p_dev={state.p_dev:.3f}, V_loss={state.V_loss:.3f}, "
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
    # ------------------------------------------------------------------
    session_context = {
        "machine_id": "PROBAT-P12",
        "coffee_id": "RW-SH1",
        "operator_id": "SIMONE",
        "style_profile": "filter_clarity",
        "brew_method": "filter",
        "batch_size_kg": 6.0,
        "charge_temp_c": 205.0,
        "drop_temp_c": 205.0,
        "duration_s": 570,
        "ambient_temp_c": 21.0,
        "ambient_rh_pct": 48.0,
        "intent_clarity": 0.90,
        "intent_sweetness": 0.75,
        "intent_body": 0.35,
        "intent_bitterness": 0.15,
        "timestamp_start": "2026-03-01T10:00:00",
    }

    # ------------------------------------------------------------------
    # 4. Coffee context
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
    # 5. Candidate sequences
    # ------------------------------------------------------------------
    candidate_control_sequences = build_candidate_control_sequences()
    print(f"\nGenerated {len(candidate_control_sequences)} candidate control sequences.")

    # ------------------------------------------------------------------
    # 6. Controller
    # ------------------------------------------------------------------
    controller = RoastController(model_dir=project_root / "artifacts" / "models")

    best, evaluations = controller.choose_best_option(
        initial_state=state,
        candidate_control_sequences=candidate_control_sequences,
        target_flavor=target_flavor,
        session_context=session_context,
        coffee_context=coffee_context,
    )

    # ------------------------------------------------------------------
    # 7. Print evaluations
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

    # ------------------------------------------------------------------
    # 9. Plot trajectories
    # ------------------------------------------------------------------
    plot_path = project_root / "artifacts" / "candidate_trajectories.png"
    plot_candidate_trajectories(
        evaluations,
        dt_s=2.0,
        save_path=plot_path,
        show=True,
    )
    print(f"\nSaved candidate trajectory plot to: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()