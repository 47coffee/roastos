from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from roastos.types import RoastState


def _extract_series(states: list[RoastState]) -> dict[str, list[float]]:
    return {
        "Tb": [s.Tb for s in states],
        "E_drum": [s.E_drum for s in states],
        "p_dry": [s.p_dry for s in states],
        "p_mai": [s.p_mai for s in states],
        "p_dev": [s.p_dev for s in states],
        "V_loss": [s.V_loss for s in states],
        "S_struct": [s.S_struct for s in states],
    }


def plot_candidate_trajectories(
    evaluations: list[Any],
    *,
    dt_s: float = 2.0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the simulated future trajectories for all candidate options.

    Expected each evaluation to have:
    - trajectory_states: list[RoastState]
    - cost: float
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.flatten()

    metric_order = [
        "Tb",
        "E_drum",
        "p_dry",
        "p_mai",
        "p_dev",
        "V_loss",
        "S_struct",
    ]

    metric_titles = {
        "Tb": "Bean Temperature",
        "E_drum": "Drum Energy",
        "p_dry": "Drying Progress",
        "p_mai": "Maillard Progress",
        "p_dev": "Development Progress",
        "V_loss": "Volatile Loss",
        "S_struct": "Structural Transformation",
    }

    metric_ylabels = {
        "Tb": "°C",
        "E_drum": "norm",
        "p_dry": "0-1",
        "p_mai": "0-1",
        "p_dev": "0-1",
        "V_loss": "index",
        "S_struct": "index",
    }

    for idx, evaluation in enumerate(evaluations, start=1):
        states = evaluation.trajectory_states
        series = _extract_series(states)
        t = [i * dt_s for i in range(len(states))]
        label = f"Opt {idx} | cost={evaluation.cost:.4f}"

        for ax_idx, metric in enumerate(metric_order):
            axes[ax_idx].plot(t, series[metric], label=label)

    for ax_idx, metric in enumerate(metric_order):
        axes[ax_idx].set_title(metric_titles[metric])
        axes[ax_idx].set_xlabel("Time (s)")
        axes[ax_idx].set_ylabel(metric_ylabels[metric])
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend(fontsize=8)

    # Hide unused subplot
    if len(axes) > len(metric_order):
        axes[-1].axis("off")

    fig.suptitle("RoastOS Candidate Future Trajectories", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)