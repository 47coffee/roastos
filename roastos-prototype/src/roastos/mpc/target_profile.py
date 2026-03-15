from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TargetPoint:
    bt: float
    et: float
    phase: str


@dataclass(frozen=True)
class TerminalTargets:
    drop_bt: Optional[float] = None
    drop_weight_kg: Optional[float] = None


@dataclass(frozen=True)
class TargetTrajectory:
    points: List[TargetPoint]
    terminal: Optional[TerminalTargets] = None
    flavour_intent: Optional[Dict[str, float]] = None
    flavour_weights: Optional[Dict[str, float]] = None

    def slice(self, horizon_steps: int) -> list[TargetPoint]:
        return self.points[:horizon_steps]

