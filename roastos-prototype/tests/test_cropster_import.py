from pathlib import Path

import pandas as pd

from roastos.data import cropster_import


def _workspace_sandbox_dir(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "test_sandbox" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_config_resolves_default_path_from_outside_project_root(monkeypatch):
    monkeypatch.chdir(_workspace_sandbox_dir("config_resolution"))

    cfg = cropster_import.load_config()

    assert cfg["data"]["raw_roast_folder"] == "data/cropster_raw/roasts"


def test_run_import_from_config_resolves_data_paths_from_project_root(monkeypatch):
    monkeypatch.chdir(_workspace_sandbox_dir("data_resolution"))
    captured_paths = {}

    def fake_import_roasts(folder):
        captured_paths["roast_folder"] = Path(folder)
        return pd.DataFrame(), pd.DataFrame()

    def fake_import_qc(folder):
        captured_paths["qc_folder"] = Path(folder)
        return pd.DataFrame(), pd.DataFrame()

    def fake_save(processed_folder, roast_sessions, roast_timeseries, qc_sessions, qc_evaluators):
        captured_paths["processed_folder"] = Path(processed_folder)

    monkeypatch.setattr(cropster_import, "import_cropster_roast_folder", fake_import_roasts)
    monkeypatch.setattr(cropster_import, "import_cropster_qc_folder", fake_import_qc)
    monkeypatch.setattr(cropster_import, "save_processed_tables", fake_save)

    cropster_import.run_import_from_config()

    project_root = Path(__file__).resolve().parents[1]
    assert captured_paths["roast_folder"] == project_root / "data" / "cropster_raw" / "roasts"
    assert captured_paths["qc_folder"] == project_root / "data" / "cropster_raw" / "qc"
    assert captured_paths["processed_folder"] == project_root / "data" / "processed"
