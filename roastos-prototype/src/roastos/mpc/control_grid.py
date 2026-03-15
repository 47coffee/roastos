from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from roastos.config import load_settings
from roastos.simulator.sim_types import RoastControl


@dataclass(frozen=True)
class CandidateSequence:
    controls: list[RoastControl]
    gas_move: float
    pressure_move: float


def build_blocked_control_sequences(
    current_gas: float,
    current_pressure: float,
    drum_speed: float,
) -> list[CandidateSequence]:
    """
    Build simple blocked control candidates around the current operating point.

    V4.0 design:
    - optimize gas + pressure
    - keep drum speed fixed
    - use constant control over the full horizon
    """
    settings = load_settings()

    gas_moves = [-0.08, -0.04, 0.0, 0.04, 0.08]
    pressure_moves = [-15.0, -7.5, 0.0, 7.5, 15.0]

    candidates: list[CandidateSequence] = []

    for dg, dp in product(gas_moves, pressure_moves):
        gas = min(max(current_gas + dg, settings.mpc.gas_min), settings.mpc.gas_max)
        pressure = min(max(current_pressure + dp, settings.mpc.pressure_min), settings.mpc.pressure_max)

        controls = [
            RoastControl(gas=gas, pressure=pressure, drum_speed=drum_speed)
            for _ in range(settings.mpc.horizon_steps)
        ]
        candidates.append(
            CandidateSequence(
                controls=controls,
                gas_move=float(dg),
                pressure_move=float(dp),
            )
        )

    return candidates
