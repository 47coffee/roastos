from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd

from roastos.config import load_settings


START_WEIGHT_CANDIDATES = [
    "start_weight_kg",
    "charge_weight_kg",
    "batch_weight_kg",
    "green_weight_kg",
    "input_weight_kg",
    "weight_kg",
    "batch_size_kg",
    "batch_kg",
]

BEAN_START_TEMP_CANDIDATES = [
    "bean_start_temp_c",
    "bean_temp_at_charge_c",
    "charge_bean_temp_c",
    "green_temp_c",
]

CHARGE_TEMP_SESSION_CANDIDATES = [
    "charge_temp_c",
    "machine_charge_temp_c",
    "charge_et_c",
    "roaster_charge_temp_c",
    "drum_charge_temp_c",
]

DROP_BT_CANDIDATES = [
    "drop_temp_c",
    "drop_bt_c",
    "drop_temperature_c",
    "end_temp_c",
]

DROP_WEIGHT_CANDIDATES = [
    "drop_weight_kg",
    "end_weight_kg",
    "roasted_weight_kg",
    "out_weight_kg",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _time_to_seconds(value):
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
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return float(value)
        except Exception:
            return None

    return None


def load_processed_data(processed_folder: str | Path | None = None):
    settings = load_settings()

    processed_folder = (
        _resolve_project_path(processed_folder)
        if processed_folder is not None
        else settings.paths.processed_folder
    )

    roast_sessions_path = processed_folder / "roast_sessions.parquet"
    roast_timeseries_path = processed_folder / "roast_timeseries.parquet"

    if not roast_sessions_path.exists():
        raise FileNotFoundError(f"Missing file: {roast_sessions_path}")
    if not roast_timeseries_path.exists():
        raise FileNotFoundError(f"Missing file: {roast_timeseries_path}")

    roast_sessions = pd.read_parquet(roast_sessions_path)
    roast_timeseries = pd.read_parquet(roast_timeseries_path)

    qc_sessions_path = processed_folder / "qc_sessions.parquet"
    qc_sessions = pd.read_parquet(qc_sessions_path) if qc_sessions_path.exists() else None

    return roast_sessions, roast_timeseries, qc_sessions


def forward_fill_machine_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    cols_to_ffill = [
        col for col in ["gas_pct", "airflow_pct", "drum_speed_pct", "drum_pressure_pa", "et_c"] if col in out.columns
    ]

    if cols_to_ffill:
        out[cols_to_ffill] = out.groupby("roast_id")[cols_to_ffill].ffill()

    return out


def derive_charge_temp_from_timeseries(roast_timeseries: pd.DataFrame) -> pd.DataFrame:
    if "et_c" not in roast_timeseries.columns:
        return pd.DataFrame(columns=["roast_id", "charge_temp_c"])

    df = roast_timeseries[["roast_id", "time_s", "et_c"]].copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["et_c"] = pd.to_numeric(df["et_c"], errors="coerce")
    df = df.dropna(subset=["roast_id", "time_s", "et_c"]).sort_values(["roast_id", "time_s"])

    rows = []

    for roast_id, grp in df.groupby("roast_id"):
        grp = grp.sort_values("time_s").reset_index(drop=True)

        pre_or_at_charge = grp[grp["time_s"] <= 0]
        if not pre_or_at_charge.empty:
            charge_temp = float(pre_or_at_charge.iloc[-1]["et_c"])
        else:
            grp["abs_time"] = grp["time_s"].abs()
            charge_temp = float(grp.sort_values("abs_time").iloc[0]["et_c"])

        rows.append({"roast_id": roast_id, "charge_temp_c": charge_temp})

    return pd.DataFrame(rows)


def align_roast_start(timeseries: pd.DataFrame) -> pd.DataFrame:
    df = timeseries.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["roast_id", "time_s"]).copy()

    out_parts = []

    for roast_id, grp in df.groupby("roast_id"):
        grp = grp.sort_values("time_s").reset_index(drop=True)
        non_negative = grp[grp["time_s"] >= 0]
        if not non_negative.empty:
            start_idx = non_negative.index[0]
            aligned = grp.loc[start_idx:].copy()
        else:
            aligned = grp.copy()

        out_parts.append(aligned)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["roast_id", "time_s"]).reset_index(drop=True)
    return out


def compute_ror(timeseries: pd.DataFrame) -> pd.DataFrame:
    df = timeseries.copy()
    df = df.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    dt = df.groupby("roast_id")["time_s"].diff()
    dbt = df.groupby("roast_id")["bt_c"].diff()

    df["ror"] = (dbt / dt.replace(0, pd.NA)) * 60.0
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


def add_roast_context(
    dataset: pd.DataFrame,
    roast_sessions: pd.DataFrame,
    roast_timeseries: pd.DataFrame,
) -> pd.DataFrame:
    rs = roast_sessions.copy()
    merge_df = rs[["roast_id"]].copy()

    weight_col = _first_existing(rs, START_WEIGHT_CANDIDATES)
    bean_start_temp_col = _first_existing(rs, BEAN_START_TEMP_CANDIDATES)
    charge_temp_session_col = _first_existing(rs, CHARGE_TEMP_SESSION_CANDIDATES)
    drop_bt_col = _first_existing(rs, DROP_BT_CANDIDATES)
    drop_weight_col = _first_existing(rs, DROP_WEIGHT_CANDIDATES)

    if weight_col is not None:
        merge_df["start_weight_kg"] = pd.to_numeric(rs[weight_col], errors="coerce")

    if bean_start_temp_col is not None:
        merge_df["bean_start_temp_c"] = pd.to_numeric(rs[bean_start_temp_col], errors="coerce")

    if charge_temp_session_col is not None:
        merge_df["charge_temp_c"] = pd.to_numeric(rs[charge_temp_session_col], errors="coerce")

    if drop_bt_col is not None:
        merge_df["actual_drop_bt"] = pd.to_numeric(rs[drop_bt_col], errors="coerce")

    if drop_weight_col is not None:
        merge_df["actual_drop_weight_kg"] = pd.to_numeric(rs[drop_weight_col], errors="coerce")

    out = dataset.merge(merge_df, on="roast_id", how="left")

    charge_from_ts = derive_charge_temp_from_timeseries(roast_timeseries)
    if not charge_from_ts.empty:
        out = out.merge(charge_from_ts, on="roast_id", how="left", suffixes=("", "_derived"))

        if "charge_temp_c_derived" in out.columns:
            if "charge_temp_c" not in out.columns:
                out["charge_temp_c"] = out["charge_temp_c_derived"]
            else:
                out["charge_temp_c"] = out["charge_temp_c"].fillna(out["charge_temp_c_derived"])
            out = out.drop(columns=["charge_temp_c_derived"])

    if "bean_start_temp_c" in out.columns and "start_temp_c" not in out.columns:
        out["start_temp_c"] = out["bean_start_temp_c"]

    return out


def add_calibration_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset = dataset.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    dataset["bt_next"] = dataset.groupby("roast_id")["bt_c"].shift(-1)
    dataset["bt_delta"] = dataset["bt_next"] - dataset["bt_c"]

    dataset["et_next"] = dataset.groupby("roast_id")["et_c"].shift(-1)
    dataset["et_step"] = dataset["et_next"] - dataset["et_c"]

    dataset["et_delta"] = dataset["et_c"] - dataset["bt_c"]

    dataset["gas_lag1"] = dataset.groupby("roast_id")["gas"].shift(1)
    dataset["airflow_lag1"] = dataset.groupby("roast_id")["airflow"].shift(1)
    dataset["pressure_lag1"] = dataset.groupby("roast_id")["pressure"].shift(1)
    dataset["et_delta_lag1"] = dataset.groupby("roast_id")["et_delta"].shift(1)
    dataset["et_c_lag1"] = dataset.groupby("roast_id")["et_c"].shift(1)

    dataset["gas_delta"] = dataset["gas"] - dataset["gas_lag1"]
    dataset["airflow_delta"] = dataset["airflow"] - dataset["airflow_lag1"]
    dataset["pressure_delta"] = dataset["pressure"] - dataset["pressure_lag1"]

    dataset["bt_c_norm"] = dataset["bt_c"] / 200.0
    dataset["et_c_norm"] = dataset["et_c"] / 250.0

    roast_duration = dataset.groupby("roast_id")["time_s"].transform("max")
    dataset["time_frac"] = dataset["time_s"] / roast_duration.replace(0, pd.NA)

    dataset = dataset.dropna(subset=["bt_next", "et_next"]).copy()
    return dataset


def build_calibration_dataset(processed_folder: str | Path | None = None):
    roast_sessions, roast_timeseries, qc_sessions = load_processed_data(processed_folder)

    full_ts = roast_timeseries.copy()
    full_ts = full_ts.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    df = roast_timeseries.copy()
    df = df.sort_values(["roast_id", "time_s"]).reset_index(drop=True)

    # Canonical control mappings
    if "gas_pct" in df.columns:
        df["gas"] = pd.to_numeric(df["gas_pct"], errors="coerce") / 100.0
    else:
        df["gas"] = pd.NA

    if "airflow_pct" in df.columns:
        df["airflow"] = pd.to_numeric(df["airflow_pct"], errors="coerce") / 100.0
    else:
        df["airflow"] = pd.NA

    if "drum_speed_pct" in df.columns:
        df["drum_speed"] = pd.to_numeric(df["drum_speed_pct"], errors="coerce") / 100.0
    else:
        df["drum_speed"] = pd.NA

    if "drum_pressure_pa" in df.columns:
        df["pressure"] = pd.to_numeric(df["drum_pressure_pa"], errors="coerce")
    else:
        df["pressure"] = pd.NA

    if "et_c" in df.columns:
        df["et_c"] = pd.to_numeric(df["et_c"], errors="coerce")

    if "bt_c" in df.columns:
        df["bt_c"] = pd.to_numeric(df["bt_c"], errors="coerce")

    df = forward_fill_machine_channels(df)
    df = align_roast_start(df)
    df = compute_ror(df)
    df = add_roast_phase(df, roast_sessions)
    df = df.dropna(subset=["bt_c"]).copy()

    wanted_cols = [
        "roast_id",
        "time_s",
        "bt_c",
        "et_c",
        "ror",
        "gas_pct",
        "airflow_pct",
        "drum_speed_pct",
        "drum_pressure_pa",
        "gas",
        "airflow",
        "drum_speed",
        "pressure",
        "phase",
        "event",
        "source_file",
    ]

    dataset = df[[c for c in wanted_cols if c in df.columns]].copy()
    dataset = add_calibration_features(dataset)
    dataset = add_roast_context(dataset, roast_sessions, full_ts)

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
            dataset = dataset.merge(qc_sessions[sensory_cols], on="roast_id", how="left")

    return dataset


def save_calibration_dataset(
    processed_folder: str | Path | None = None,
    output_path: str | Path | None = None,
):
    settings = load_settings()
    dataset = build_calibration_dataset(processed_folder)

    if output_path is None:
        output_path = settings.paths.calibration_dataset
    else:
        output_path = _resolve_project_path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print("Calibration dataset saved.")
    print("Output:", output_path)
    print("Rows:", len(dataset))
    print("Roasts:", dataset["roast_id"].nunique())
    print("Columns:", list(dataset.columns))
    print("Missing values summary:")

    tracked = [
        c for c in [
            "bt_c", "et_c", "gas_pct", "airflow_pct", "drum_speed_pct", "drum_pressure_pa",
            "gas", "airflow", "drum_speed", "pressure", "ror",
            "start_weight_kg", "bean_start_temp_c", "charge_temp_c",
            "actual_drop_bt", "actual_drop_weight_kg",
        ]
        if c in dataset.columns
    ]
    print(dataset[tracked].isna().sum())


if __name__ == "__main__":
    save_calibration_dataset()
