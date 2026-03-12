from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

"""This module implements the data import pipeline for Cropster roast and QC files. It defines 
functions to read Excel files exported from Cropster, extract relevant metadata and timeseries data, 
and save the processed data in a structured format (Parquet) for further analysis. 
The main entry point is the run_import_from_config function, which reads configuration parameters from a 
config file, processes all roast and QC files in the specified folders, and saves 
the outputs to a designated processed data folder. The code includes robust
 handling of different sheet formats and column naming conventions commonly found in Cropster exports."""

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/roastos.ini")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config_path = Path(config_path)
    candidates = []

    if config_path.is_absolute():
        candidates.append(config_path)
    else:
        candidates.extend(
            [
                Path.cwd() / config_path,
                _project_root() / config_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find config file '{config_path}'. Checked: {candidate_list}"
    )


def _resolve_project_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    base_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (base_dir / path).resolve()


def load_config(config_path: str | Path = "config/roastos.ini") -> ConfigParser:
    resolved_config_path = _resolve_config_path(config_path)
    cfg = ConfigParser()
    cfg.read(resolved_config_path)
    return cfg


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _safe_sheet(workbook: dict[str, pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    return workbook.get(name)


def _read_excel_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None)


def _normalize_curve_sheet(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Expect Cropster curve sheets in shape:
    Time (s) | Value
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_s", value_name])

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Try to identify time column
    time_col = None
    value_col = None

    for c in out.columns:
        cl = c.lower()
        if "time" in cl:
            time_col = c
            break

    for c in out.columns:
        if c != time_col:
            value_col = c
            break

    if time_col is None or value_col is None:
        return pd.DataFrame(columns=["time_s", value_name])

    out = out[[time_col, value_col]].copy()
    out.columns = ["time_s", value_name]
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.dropna(subset=["time_s"]).sort_values("time_s")
    return out.reset_index(drop=True)


def _extract_first_value(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    return df.iloc[0][col]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# ------------------------------------------------------------
# Roast import
# ------------------------------------------------------------

def parse_cropster_roast_file(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - roast_session_df: one row
    - roast_timeseries_df: one row per time point
    """
    path = Path(path)
    wb = _read_excel_all_sheets(path)

    general = _safe_sheet(wb, "General")
    comments = _safe_sheet(wb, "Comments")

    general = _clean_columns(general) if general is not None else pd.DataFrame()
    comments = _clean_columns(comments) if comments is not None else pd.DataFrame()

    # --- metadata ---
    def _find_column(df, candidates):

        for c in df.columns:
            name = c.strip().lower()

            for candidate in candidates:
                if candidate in name:
                    return c

        return None

    # detect roast id column
    roast_id_col = _find_column(
        general,
        ["id tag", "roast id", "batch id", "idtag"]
    )

    if roast_id_col is not None:
        roast_id = general.iloc[0][roast_id_col]
    else:
        roast_id = None
    
    if roast_id is None or pd.isna(roast_id):
        roast_id = path.stem
    print("Detected roast_id:", roast_id)
    #roast_id = _extract_first_value(general, "ID Tag")
    roast_name = _extract_first_value(general, "Roast name")
    roast_date = _extract_first_value(general, "Roast date")
    profile = _extract_first_value(general, "Profile")
    machine = _extract_first_value(general, "Machine")
    start_weight = _extract_first_value(general, "Start weight")
    end_weight = _extract_first_value(general, "End weight")
    weight_loss_pct = _extract_first_value(general, "Weight loss")
    duration = _extract_first_value(general, "Duration")
    start_temp = _extract_first_value(general, "Start temp")
    end_temp = _extract_first_value(general, "End temp")
    dev_time = _extract_first_value(general, "Dev. time")
    dev_ratio = _extract_first_value(general, "Dev. time ratio")
    first_crack = _extract_first_value(general, "First crack")
    roast_value = _extract_first_value(general, "Roast value")
    sensorial = _extract_first_value(general, "Sensorial")
    notes = _extract_first_value(general, "Notes")

    roast_session_df = pd.DataFrame(
        [
            {
                "roast_id": roast_id,
                "roast_name": roast_name,
                "roast_date": roast_date,
                "profile": profile,
                "machine": machine,
                "start_weight": start_weight,
                "end_weight": end_weight,
                "weight_loss_pct": weight_loss_pct,
                "duration_s": duration,
                "start_temp_c": start_temp,
                "end_temp_c": end_temp,
                "dev_time_s": dev_time,
                "dev_ratio": dev_ratio,
                "first_crack_s": first_crack,
                "roast_value": roast_value,
                "sensorial": sensorial,
                "notes": notes,
                "source_file": path.name,
            }
        ]
    )

    # --- event comments ---
    event_df = pd.DataFrame(columns=["time_s", "event"])
    if comments is not None and not comments.empty:
        c = comments.copy()
        c.columns = [str(x).strip() for x in c.columns]
        # Try flexible detection
        time_col = None
        event_col = None
        for col in c.columns:
            cl = col.lower()
            if "time" in cl:
                time_col = col
            if "comment" in cl or "event" in cl or "label" in cl:
                event_col = col
        if time_col and event_col:
            event_df = c[[time_col, event_col]].copy()
            event_df.columns = ["time_s", "event"]
            event_df["time_s"] = pd.to_numeric(event_df["time_s"], errors="coerce")
            event_df = event_df.dropna(subset=["time_s"])

    # --- curves ---
    sheet_map = {
        "Curve - Bean temperature": "bt_c",
        "Curve - Exhaust temperature": "et_c",
        "Curve - Drum temperature": "dt_c",
        "Curve - Inlet temperature": "it_c",
        "Curve - Gas": "gas_pct",
        "Curve - Gas control": "gas_control_pct",
        "Curve - Airflow": "airflow_pct",
        "Curve - Airflow control": "airflow_control_pct",
        "Curve - Drum pressure": "drum_pressure_pa",
        "Curve - Drum speed": "drum_speed_pct",
        "Curve - drumSpeedControl": "drum_speed_control_pct",
    }

    merged = None

    for sheet_name, value_name in sheet_map.items():
        sheet_df = _safe_sheet(wb, sheet_name)
        curve_df = _normalize_curve_sheet(sheet_df, value_name)
        if merged is None:
            merged = curve_df
        else:
            merged = pd.merge(merged, curve_df, on="time_s", how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=["time_s"])

    merged = merged.sort_values("time_s").reset_index(drop=True)

    # attach events
    if not event_df.empty:
        merged = pd.merge(merged, event_df, on="time_s", how="left")
    else:
        merged["event"] = None

    merged["roast_id"] = roast_id
    merged["source_file"] = path.name

    # Optional: derive RoR from BT if not directly present
    if "bt_c" in merged.columns:
        bt = pd.to_numeric(merged["bt_c"], errors="coerce")
        t = pd.to_numeric(merged["time_s"], errors="coerce")
        if bt.notna().sum() >= 3:
            merged["ror_c_per_min"] = (
                bt.interpolate().diff() / t.interpolate().diff()
            ) * 60.0
        else:
            merged["ror_c_per_min"] = None

    return roast_session_df, merged


# ------------------------------------------------------------
# QC / sensory import
# ------------------------------------------------------------

def parse_cropster_qc_file(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - qc_sessions_df: one row per QC file / session
    - qc_evaluators_df: one row per evaluator if available
    """
    path = Path(path)
    wb = _read_excel_all_sheets(path)

    general = _clean_columns(_safe_sheet(wb, "General")) if _safe_sheet(wb, "General") is not None else pd.DataFrame()
    categories_sheet = None
    per_eval_sheet = None

    for sname in wb.keys():
        if sname.startswith("Categories"):
            categories_sheet = _clean_columns(wb[sname])
        if sname.startswith("Per evaluator"):
            per_eval_sheet = _clean_columns(wb[sname])

    roast_id = _extract_first_value(general, "Lot ID-Tag")
    lot_name = _extract_first_value(general, "Lot name")
    qc_id = _extract_first_value(general, "QC ID-Tag")
    qc_label = _extract_first_value(general, "QC label")
    analysis_date = _extract_first_value(general, "Sensorial analysis date")
    lab = _extract_first_value(general, "Lab")
    evaluators = _extract_first_value(general, "Evaluators")
    evaluator_count = _extract_first_value(general, "# Evaluators")
    sens_descriptors = _extract_first_value(general, "Sens. descriptors")
    final_score_general = _extract_first_value(general, "Final score")
    processing = _extract_first_value(general, "Processing")
    crop_year = _extract_first_value(general, "Crop year")
    varieties = _extract_first_value(general, "Varieties")
    country = _extract_first_value(general, "Country")
    moisture = _extract_first_value(general, "Moisture")
    water_activity = _extract_first_value(general, "Water activity")
    density = _extract_first_value(general, "Density")

    # Use Categories sheet for canonical sensory values if present
    row = categories_sheet.iloc[0] if categories_sheet is not None and not categories_sheet.empty else pd.Series(dtype=object)

    qc_sessions_df = pd.DataFrame(
        [
            {
                "roast_id": roast_id,
                "lot_name": lot_name,
                "qc_id": qc_id,
                "qc_label": qc_label,
                "analysis_date": analysis_date,
                "lab": lab,
                "evaluators": evaluators,
                "evaluator_count": evaluator_count,
                "final_score": row.get("Final score", final_score_general),
                "fragrance": row.get("Fragrance"),
                "aroma": row.get("Aroma"),
                "flavor": row.get("Flavor"),
                "aftertaste": row.get("Aftertaste"),
                "acidity": row.get("Acidity"),
                "sweetness": row.get("Sweetness"),
                "mouthfeel": row.get("Mouthfeel"),
                "overall": row.get("Overall"),
                "non_uniform_cups": row.get("Non-Uniform Cups"),
                "defective_cups": row.get("Defective Cups"),
                "general_descriptors": row.get("General Descriptors Descriptors", sens_descriptors),
                "processing": processing,
                "crop_year": crop_year,
                "varieties": varieties,
                "country": country,
                "green_moisture": moisture,
                "green_water_activity": water_activity,
                "green_density": density,
                "source_file": path.name,
            }
        ]
    )

    # evaluator-level table
    qc_evaluators_df = pd.DataFrame()
    if per_eval_sheet is not None and not per_eval_sheet.empty:
        out = per_eval_sheet.copy()
        out["roast_id"] = roast_id
        out["qc_id"] = qc_id
        out["source_file"] = path.name
        qc_evaluators_df = out

    return qc_sessions_df, qc_evaluators_df


# ------------------------------------------------------------
# Folder import
# ------------------------------------------------------------

def import_cropster_roast_folder(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(folder)
    session_frames = []
    curve_frames = []

    for path in sorted(folder.glob("*.xlsx")):
        try:
            s_df, c_df = parse_cropster_roast_file(path)
            session_frames.append(s_df)
            curve_frames.append(c_df)
        except Exception as e:
            print(f"[WARN] Failed to parse roast file {path.name}: {e}")

    sessions = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    return sessions, curves


def import_cropster_qc_folder(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(folder)
    qc_session_frames = []
    qc_eval_frames = []

    for path in sorted(folder.glob("*.xlsx")):
        try:
            s_df, e_df = parse_cropster_qc_file(path)
            qc_session_frames.append(s_df)
            if e_df is not None and not e_df.empty:
                qc_eval_frames.append(e_df)
        except Exception as e:
            print(f"[WARN] Failed to parse QC file {path.name}: {e}")

    qc_sessions = pd.concat(qc_session_frames, ignore_index=True) if qc_session_frames else pd.DataFrame()
    qc_evaluators = pd.concat(qc_eval_frames, ignore_index=True) if qc_eval_frames else pd.DataFrame()
    return qc_sessions, qc_evaluators


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

def save_processed_tables(
    processed_folder: str | Path,
    roast_sessions: pd.DataFrame,
    roast_timeseries: pd.DataFrame,
    qc_sessions: pd.DataFrame,
    qc_evaluators: pd.DataFrame,
) -> None:
    processed_folder = Path(processed_folder)
    processed_folder.mkdir(parents=True, exist_ok=True)

    roast_sessions.to_parquet(processed_folder / "roast_sessions.parquet", index=False)
    roast_timeseries.to_parquet(processed_folder / "roast_timeseries.parquet", index=False)
    qc_sessions.to_parquet(processed_folder / "qc_sessions.parquet", index=False)

    if qc_evaluators is not None and not qc_evaluators.empty:
        qc_evaluators.to_parquet(processed_folder / "qc_evaluators.parquet", index=False)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def run_import_from_config(config_path: str | Path = "config/roastos.ini") -> None:
    resolved_config_path = _resolve_config_path(config_path)
    cfg = load_config(resolved_config_path)

    if "data" not in cfg:
        raise KeyError(
            f"Missing [data] section in config file: {resolved_config_path}"
        )

    required_keys = ("raw_roast_folder", "raw_qc_folder", "processed_folder")
    missing_keys = [key for key in required_keys if key not in cfg["data"]]
    if missing_keys:
        missing_keys_text = ", ".join(missing_keys)
        raise KeyError(
            f"Missing keys in [data] section of {resolved_config_path}: {missing_keys_text}"
        )

    roast_folder = _resolve_project_path(cfg["data"]["raw_roast_folder"], resolved_config_path)
    qc_folder = _resolve_project_path(cfg["data"]["raw_qc_folder"], resolved_config_path)
    processed_folder = _resolve_project_path(cfg["data"]["processed_folder"], resolved_config_path)

    roast_sessions, roast_timeseries = import_cropster_roast_folder(roast_folder)
    qc_sessions, qc_evaluators = import_cropster_qc_folder(qc_folder)

    save_processed_tables(
        processed_folder=processed_folder,
        roast_sessions=roast_sessions,
        roast_timeseries=roast_timeseries,
        qc_sessions=qc_sessions,
        qc_evaluators=qc_evaluators,
    )

    print("Cropster import complete.")
    print(f"Roast sessions:   {len(roast_sessions)}")
    print(f"Roast timeseries: {len(roast_timeseries)}")
    print(f"QC sessions:      {len(qc_sessions)}")
    print(f"QC evaluators:    {len(qc_evaluators)}")


if __name__ == "__main__":
    run_import_from_config()
