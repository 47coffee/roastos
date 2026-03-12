from pathlib import Path
from datetime import time
import pandas as pd


def load_processed_data(processed_folder="data/processed"):
    processed_folder = Path(processed_folder)

    roast_sessions = pd.read_parquet(processed_folder / "roast_sessions.parquet")
    roast_timeseries = pd.read_parquet(processed_folder / "roast_timeseries.parquet")

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

    # already numeric
    if isinstance(value, (int, float)):
        return float(value)

    # pandas timedelta
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()

    # datetime.time
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second

    # string formats
    if isinstance(value, str):
        value = value.strip()

        # try HH:MM:SS
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


def compute_ror(timeseries):
    df = timeseries.copy()

    df["ror"] = (
        df.groupby("roast_id")["bt_c"]
        .diff()
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


def add_roast_phase(timeseries, roast_sessions):
    df = timeseries.copy()
    rs = roast_sessions.copy()

    # Convert first crack to numeric seconds
    rs["first_crack_s_numeric"] = rs["first_crack_s"].apply(_time_to_seconds)

    fc_times = rs.set_index("roast_id")["first_crack_s_numeric"].to_dict()

    phases = []
    for _, row in df.iterrows():
        roast_id = row["roast_id"]
        fc_time_s = fc_times.get(roast_id)
        phases.append(classify_phase(row, fc_time_s))

    df["phase"] = phases
    return df


def build_calibration_dataset():
    roast_sessions, roast_timeseries, qc_sessions = load_processed_data()

    df = roast_timeseries.copy()

    # compute RoR
    df = compute_ror(df)

    # phase classification
    df = add_roast_phase(df, roast_sessions)

    # remove bad rows
    df = df.dropna(subset=["bt_c"])

    # normalize controls
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

    # keep only existing columns
    dataset = df[[c for c in wanted_cols if c in df.columns]].copy()

    # attach sensory if available
    if qc_sessions is not None and not qc_sessions.empty:
        sensory_cols = [
            c for c in [
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


def save_calibration_dataset():
    dataset = build_calibration_dataset()

    output_path = Path("data/processed/calibration_dataset.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(output_path, index=False)

    print("Calibration dataset saved.")
    print("Rows:", len(dataset))
    print("Roasts:", dataset["roast_id"].nunique())
    print("Columns:", list(dataset.columns))


if __name__ == "__main__":
    save_calibration_dataset()
