from datetime import time
from pathlib import Path

import pandas as pd

from roastos.data import dataset_builder


def _workspace_sandbox_dir(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "test_sandbox" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_add_roast_phase_accepts_datetime_time_first_crack():
    timeseries = pd.DataFrame(
        [
            {"roast_id": "R1", "time_s": 100},
            {"roast_id": "R1", "time_s": 200},
            {"roast_id": "R1", "time_s": 500},
        ]
    )
    roast_sessions = pd.DataFrame(
        [
            {"roast_id": "R1", "first_crack_s": time(0, 7, 30)},
        ]
    )

    result = dataset_builder.add_roast_phase(timeseries, roast_sessions)

    assert result["phase"].tolist() == ["drying", "maillard", "development"]


def test_load_processed_data_resolves_default_path_from_outside_project_root(monkeypatch):
    monkeypatch.chdir(_workspace_sandbox_dir("dataset_builder"))

    roast_sessions, roast_timeseries, qc_sessions = dataset_builder.load_processed_data()

    assert not roast_sessions.empty
    assert not roast_timeseries.empty
    assert qc_sessions is not None
