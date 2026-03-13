from __future__ import annotations

from pathlib import Path
from datetime import time

import pandas as pd


DEFAULT_PROCESSED_FOLDER = "data/processed"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(path_value: str | Path = DEFAULT_PROCESSED_FOLDER) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def load_processed_data(processed_folder: str | Path = DEFAULT_PROCESSED_FOLDER):
    processed_folder = _resolve_project_path(processed_folder)

    roast_sessions_path = processed_folder / "roast_sessions.parquet"
    roast_timeseries_path = processed_folder / "roast_timeseries.parquet"

    if not roast_sessions_path.exists():
        raise FileNotFoundError(f"Missing file: {roast_sessions_path}")
    if not roast_timeseries_path.exists():
        raise FileNotFoundError(f"Missing file: {roast_timeseries_path}")

    roast_sessions = pd.read_parquet(roast_sessions_path)
    roast_timeseries = pd.read_parquet(roast_timeseries_path)

    qc_sessions_path = processed_folder / "qc_sessions.parquet"
    if qc_sessions_path.exists():
        qc_sessions = pd.read_parquet(qc_sessions_path)
    else:
        qc_sessions = None

    return roast_sessions, roast_timeseries, qc_sessions


def _time_to_seconds(value):
    """
    Convert Cropster-like time values into seconds.

    Supports:
    - numeric seconds
    - pandas Timedelta
    - datetime.time
    - strings like HH:MM:SS
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, pd.Timedelta):
        return value.total_seconds()

    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second

    if isinstance(value, str):
        value = value.strip()
        try:
            parts = value.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            else:
                return float(value)
        except Exception:
            return None

    return None


def compute_ror(timeseries: pd.DataFrame) -> pd.DataFrame:
    df = timeseries.copy()

    df["ror"] = (
        df.groupby("roast_id")["bt_c"].diff()
        / df.groupby("roast_id")["time_s"].diff()
    ) * 60

    return df


def classify_phase(row, first_crack_time_s):
    t = row["time_s"]

    if first_crack_time_s is None:
        return "roast"

    if t < first_crack_time_s * 0.4:
        return "drying"

    if t < first_crack_time_s:
        return "maillard"

    return "development"


def add_roast_phase(timeseries: pd.DataFrame, roast_sessions: pd.DataFrame) -> pd.DataFrame:
    df = timeseries.copy()
    rs = roast_sessions.copy()

    rs["first_crack_s_numeric"] = rs["first_crack_s"].apply(_time_to_seconds)
    fc_times = rs.set_index("roast_id")["first_crack_s_numeric"].to_dict()

    phases = []
    for _, row in df.iterrows():
        roast_id = row["roast_id"]
        fc_time_s = fc_times.get(roast_id)
        phases.append(classify_phase(row, fc_time_s))

    df["phase"] = phases
    return df


def add_calibration_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    dataset["bt_next"] = dataset.groupby("roast_id")["bt_c"].shift(-1)
    dataset["bt_delta"] = dataset["bt_next"] - dataset["bt_c"]

    dataset["et_delta"] = dataset["et_c"] - dataset["bt_c"]
    dataset["gas_lag1"] = dataset.groupby("roast_id")["gas"].shift(1)
    dataset["pressure_lag1"] = dataset.groupby("roast_id")["pressure"].shift(1)
    dataset["et_delta_lag1"] = dataset.groupby("roast_id")["et_delta"].shift(1)

    # Keep only rows where the next-step target exists.
    dataset = dataset.dropna(subset=["bt_next"]).copy()

    return dataset


def build_calibration_dataset(processed_folder: str | Path = DEFAULT_PROCESSED_FOLDER):
    roast_sessions, roast_timeseries, qc_sessions = load_processed_data(processed_folder)

    df = roast_timeseries.copy()

    df = compute_ror(df)
    df = add_roast_phase(df, roast_sessions)

    df = df.dropna(subset=["bt_c"]).copy()

    if "gas_pct" in df.columns:
        df["gas"] = df["gas_pct"] / 100.0
    else:
        df["gas"] = None

    if "drum_speed_pct" in df.columns:
        df["drum_speed"] = df["drum_speed_pct"] / 100.0
    else:
        df["drum_speed"] = None

    if "drum_pressure_pa" in df.columns:
        df["pressure"] = df["drum_pressure_pa"]
    else:
        df["pressure"] = None

    wanted_cols = [
        "roast_id",
        "time_s",
        "bt_c",
        "et_c",
        "ror",
        "gas",
        "pressure",
        "drum_speed",
        "phase",
    ]

    dataset = df[[c for c in wanted_cols if c in df.columns]].copy()
    dataset = add_calibration_features(dataset)

    if qc_sessions is not None and not qc_sessions.empty:
        sensory_cols = [
            c
            for c in [
                "roast_id",
                "sweetness",
                "acidity",
                "mouthfeel",
                "overall",
                "final_score",
            ]
            if c in qc_sessions.columns
        ]

        if "roast_id" in sensory_cols:
            dataset = dataset.merge(
                qc_sessions[sensory_cols],
                on="roast_id",
                how="left",
            )

    return dataset


def save_calibration_dataset(
    processed_folder: str | Path = DEFAULT_PROCESSED_FOLDER,
    output_path: str | Path | None = None,
):
    dataset = build_calibration_dataset(processed_folder)

    if output_path is None:
        output_path = _resolve_project_path("data/processed/calibration_dataset.parquet")
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = (_project_root() / output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print("Calibration dataset saved.")
    print("Output:", output_path)
    print("Rows:", len(dataset))
    print("Roasts:", dataset["roast_id"].nunique())
    print("Columns:", list(dataset.columns))


if __name__ == "__main__":
    save_calibration_dataset()
