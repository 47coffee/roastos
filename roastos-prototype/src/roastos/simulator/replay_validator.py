from __future__ import annotations

import math
from typing import List, Optional

import pandas as pd

from .calibrated_simulator import CalibratedRoasterSimulator
from .sim_types import ReplayMetrics, ReplayResult, RoastContext, RoastControl, RoastSimState
from .state_estimator import RoastStateEstimator


CONTEXT_COLS = [
    "start_weight_kg",
    "bean_start_temp_c",
    "charge_temp_c",
    "start_temp_c",  # legacy
    "actual_drop_bt",
    "actual_drop_weight_kg",
]


def _rmse(actual: List[float], pred: List[float]) -> float:
    if not actual:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual))


def _mae(actual: List[float], pred: List[float]) -> float:
    if not actual:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def _normalize_replay_dataframe(
    df: pd.DataFrame,
    roast_id: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()

    roast_id_col = _pick_first_existing(
        df,
        ["roast_id", "session_id", "roast_uuid", "batch_id"],
        "roast id",
    )

    available_ids = []
    if roast_id_col in df.columns:
        available_ids = sorted(df[roast_id_col].dropna().astype(str).unique().tolist())

    if roast_id is not None:
        df = df[df[roast_id_col].astype(str) == str(roast_id)].copy()
        if df.empty:
            raise ValueError(
                f"No rows found after roast filtering for roast_id={roast_id}. "
                f"Available examples: {available_ids[:10]}"
            )
    else:
        if df[roast_id_col].dropna().empty:
            raise ValueError("Replay dataframe does not contain any valid roast_id values.")
        first_roast = df[roast_id_col].dropna().astype(str).iloc[0]
        df = df[df[roast_id_col].astype(str) == first_roast].copy()

    if df.empty:
        raise ValueError("Replay dataframe is empty after roast filtering.")

    df = df.reset_index(drop=True)

    bt_col = _pick_first_existing(df, ["bt_c", "bt"], "BT")
    et_col = _pick_first_existing(df, ["et_c", "et"], "ET")
    ror_col = _pick_first_existing(df, ["ror"], "RoR")
    gas_col = _pick_first_existing(df, ["gas", "gas_norm", "gas_pct"], "gas")
    pressure_col = _pick_first_existing(df, ["pressure", "drum_pressure_pa"], "pressure")
    phase_col = _pick_first_existing(df, ["phase"], "phase")
    time_col = _pick_first_existing(df, ["time_s", "seconds", "elapsed_seconds", "t_sec"], "time")

    drum_col = None
    for c in ["drum_speed", "drum", "drum_pct"]:
        if c in df.columns:
            drum_col = c
            break

    out = pd.DataFrame(index=df.index)
    out["roast_id"] = df[roast_id_col].astype(str).to_numpy()
    out["time_s"] = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    out["bt_c"] = pd.to_numeric(df[bt_col], errors="coerce").to_numpy()
    out["et_c"] = pd.to_numeric(df[et_col], errors="coerce").to_numpy()
    out["ror"] = pd.to_numeric(df[ror_col], errors="coerce").to_numpy()
    out["gas"] = pd.to_numeric(df[gas_col], errors="coerce").to_numpy()
    out["pressure"] = pd.to_numeric(df[pressure_col], errors="coerce").to_numpy()
    out["phase"] = df[phase_col].astype(str).to_numpy()

    if drum_col is not None:
        out["drum_speed"] = pd.to_numeric(df[drum_col], errors="coerce").fillna(0.65).to_numpy()
    else:
        out["drum_speed"] = 0.65

    if gas_col == "gas_pct":
        out["gas"] = out["gas"] / 100.0

    for c in CONTEXT_COLS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()

    if out["ror"].isna().any():
        rebuilt_ror = out["bt_c"].diff() / out["time_s"].diff().replace(0, pd.NA) * 60.0
        out["ror"] = out["ror"].fillna(rebuilt_ror).fillna(0.0)

    debug_cols = ["time_s", "bt_c", "et_c", "ror", "gas", "pressure", "phase", "drum_speed"]
    missing_summary = {c: int(out[c].isna().sum()) for c in debug_cols if c in out.columns}

    out = out.dropna(
        subset=["roast_id", "time_s", "bt_c", "et_c", "ror", "gas", "pressure", "phase"]
    ).reset_index(drop=True)

    if len(out) < 2:
        raise ValueError(
            f"Replay dataframe for roast_id={roast_id} contains only {len(out)} valid row(s) after cleaning. "
            f"Need at least 2. Missing-value summary before dropna: {missing_summary}"
        )

    return out


def _build_context(df: pd.DataFrame) -> RoastContext:
    first = df.iloc[0]

    bean_start_temp = None
    if "bean_start_temp_c" in df.columns:
        bean_start_temp = _safe_optional_float(first["bean_start_temp_c"])
    elif "start_temp_c" in df.columns:
        bean_start_temp = _safe_optional_float(first["start_temp_c"])

    charge_temp = _safe_optional_float(first["charge_temp_c"]) if "charge_temp_c" in df.columns else None

    return RoastContext(
        roast_id=str(first["roast_id"]),
        start_weight_kg=_safe_optional_float(first["start_weight_kg"]) if "start_weight_kg" in df.columns else None,
        bean_start_temp_c=bean_start_temp,
        charge_temp_c=charge_temp,
        start_temp_c=bean_start_temp,
        target_drop_bt=_safe_optional_float(first["actual_drop_bt"]) if "actual_drop_bt" in df.columns else None,
        target_drop_weight_kg=_safe_optional_float(first["actual_drop_weight_kg"]) if "actual_drop_weight_kg" in df.columns else None,
    )


def _reconstruct_full_latent_series(
    df: pd.DataFrame,
    simulator: CalibratedRoasterSimulator,
) -> list[float]:
    e_drum_raw = 0.0
    series = []

    for _, row in df.iterrows():
        gas_used = simulator._normalize_gas(_safe_float(row["gas"], 0.0))
        pressure_used = simulator._prepare_pressure(_safe_float(row["pressure"], 0.0))

        e_drum_raw = simulator._compute_e_drum_raw(
            prev_raw=e_drum_raw,
            gas=gas_used,
            pressure=pressure_used,
        )
        series.append(e_drum_raw)

    return series


def _print_initial_replay_debug(
    first_row: pd.Series,
    context: RoastContext,
    initial_state: RoastSimState,
) -> None:
    print("\n==============================")
    print("Initial replay state")
    print("==============================")

    print("Dataset first row used for replay")
    print("--------------------------------")
    print("time_s:", _safe_float(first_row.get("time_s"), 0.0))
    print("BT:", _safe_float(first_row.get("bt_c"), 0.0))
    print("ET:", _safe_float(first_row.get("et_c"), 0.0))
    print("RoR:", _safe_float(first_row.get("ror"), 0.0))
    print("gas:", _safe_float(first_row.get("gas"), 0.0))
    print("pressure:", _safe_float(first_row.get("pressure"), 0.0))
    print("drum_speed:", _safe_float(first_row.get("drum_speed"), 0.65))
    print("phase:", first_row.get("phase"))

    print("\nContext")
    print("--------------------------------")
    print("roast_id:", context.roast_id)
    print("start_weight_kg:", context.start_weight_kg)
    print("bean_start_temp_c:", context.bean_start_temp_c)
    print("charge_temp_c:", context.charge_temp_c)
    print("start_temp_c (legacy alias):", context.start_temp_c)
    print("target_drop_bt:", context.target_drop_bt)
    print("target_drop_weight_kg:", context.target_drop_weight_kg)

    print("\nConstructed simulator initial state")
    print("--------------------------------")
    print("BT:", initial_state.bt)
    print("ET:", initial_state.et)
    print("RoR:", initial_state.ror)
    print("e_drum_raw:", initial_state.e_drum_raw)
    print("e_drum:", initial_state.e_drum)
    print("m_burden:", initial_state.m_burden)
    print("p_dry:", initial_state.p_dry)
    print("p_mai:", initial_state.p_mai)
    print("p_dev:", initial_state.p_dev)
    print("phase:", initial_state.phase)
    print("gas:", initial_state.gas)
    print("pressure:", initial_state.pressure)
    print("drum_speed:", initial_state.drum_speed)

    print("==============================\n")


def _build_initial_state_open_loop(
    df: pd.DataFrame,
    simulator: CalibratedRoasterSimulator,
    context: RoastContext,
) -> RoastSimState:
    first = df.iloc[0]
    latent_series = _reconstruct_full_latent_series(df, simulator)
    e_drum_raw = latent_series[0]

    initial_state = simulator.build_initial_state(
        t_sec=_safe_float(first["time_s"], 0.0),
        bt=_safe_float(first["bt_c"]),
        et=_safe_float(first["et_c"]),
        ror=_safe_float(first["ror"], 0.0),
        gas=_safe_float(first["gas"], 0.0),
        pressure=_safe_float(first["pressure"], 0.0),
        drum_speed=_safe_float(first["drum_speed"], 0.65),
        e_drum_raw=e_drum_raw,
        context=context,
        phase=str(first["phase"]),
    )
    _print_initial_replay_debug(first, context, initial_state)
    return initial_state


def _build_initial_state_estimated(
    df: pd.DataFrame,
    simulator: CalibratedRoasterSimulator,
    context: RoastContext,
    estimator: RoastStateEstimator,
) -> RoastSimState:
    first = df.iloc[0]
    latent_series = _reconstruct_full_latent_series(df, simulator)
    e_drum_raw = latent_series[0]

    control0 = RoastControl(
        gas=_safe_float(first["gas"], 0.0),
        pressure=_safe_float(first["pressure"], 0.0),
        drum_speed=_safe_float(first["drum_speed"], 0.65),
    )

    initial_state = estimator.initialize(
        t_sec=_safe_float(first["time_s"], 0.0),
        measured_bt=_safe_float(first["bt_c"], 0.0),
        measured_et=_safe_float(first["et_c"], 0.0),
        measured_ror=_safe_float(first["ror"], 0.0),
        control=control0,
        e_drum_raw=e_drum_raw,
        context=context,
        phase=str(first["phase"]),
    )
    _print_initial_replay_debug(first, context, initial_state)
    return initial_state


def replay_roast_dataframe(
    df: pd.DataFrame,
    simulator: CalibratedRoasterSimulator,
    roast_id: Optional[str] = None,
    teacher_force_et: bool = True,
    teacher_force_ror: bool = True,
    teacher_force_phase: bool = True,
    warmup_rows: int = 120,
    use_estimator: bool = False,
) -> ReplayResult:
    replay_df = _normalize_replay_dataframe(df, roast_id=roast_id)
    context = _build_context(replay_df)

    estimator = RoastStateEstimator(simulator) if use_estimator else None

    if estimator is not None:
        current = _build_initial_state_estimated(
            replay_df,
            simulator=simulator,
            context=context,
            estimator=estimator,
        )
    else:
        current = _build_initial_state_open_loop(
            replay_df,
            simulator=simulator,
            context=context,
        )

    rows = []

    actual_bt = []
    pred_bt = []
    actual_et = []
    pred_et = []
    actual_ror = []
    pred_ror = []

    for i in range(len(replay_df) - 1):
        row_t = replay_df.iloc[i]
        row_next = replay_df.iloc[i + 1]

        control = RoastControl(
            gas=_safe_float(row_t["gas"], 0.0),
            pressure=_safe_float(row_t["pressure"], 0.0),
            drum_speed=_safe_float(row_t["drum_speed"], 0.65),
        )

        forced_et = _safe_float(row_t["et_c"]) if teacher_force_et else None
        forced_ror = _safe_float(row_t["ror"]) if teacher_force_ror else None
        phase_override = str(row_t["phase"]) if teacher_force_phase else None

        actual_bt_i = _safe_float(row_next["bt_c"])
        actual_et_i = _safe_float(row_next["et_c"])
        actual_ror_i = _safe_float(row_next["ror"])

        if estimator is not None:
            pred_state, corrected_state = estimator.estimate_next(
                state=current,
                control=control,
                measured_bt_next=actual_bt_i,
                measured_et_next=actual_et_i,
                measured_ror_next=actual_ror_i,
                context=context,
                teacher_forced_et=forced_et,
                teacher_forced_ror=forced_ror,
                phase_override=phase_override,
            )
            current = corrected_state
        else:
            result = simulator.step(
                current,
                control,
                teacher_forced_et=forced_et,
                phase_override=phase_override,
                teacher_forced_ror=forced_ror,
                context=context,
            )
            pred_state = result.next_state
            current = pred_state

        actual_bt.append(actual_bt_i)
        pred_bt.append(pred_state.bt)

        if not teacher_force_et:
            actual_et.append(actual_et_i)
            pred_et.append(pred_state.et)

        actual_ror.append(actual_ror_i)
        pred_ror.append(pred_state.ror)

        rows.append(
            {
                "i": i + 1,
                "roast_id": str(row_t["roast_id"]),
                "time_s": _safe_float(row_next["time_s"]),
                "phase_used": pred_state.phase,
                "phase_actual_next": str(row_next["phase"]),
                "gas": control.gas,
                "pressure": control.pressure,
                "drum_speed": control.drum_speed,
                "start_weight_kg": context.start_weight_kg,
                "bean_start_temp_c": context.bean_start_temp_c,
                "charge_temp_c": context.charge_temp_c,
                "start_temp_c": context.bean_start_temp_c,
                "observer_enabled": bool(estimator is not None),
                "actual_bt": actual_bt_i,
                "pred_bt": pred_state.bt,
                "actual_et": actual_et_i,
                "pred_et": pred_state.et,
                "actual_ror": actual_ror_i,
                "pred_ror": pred_state.ror,
                "pred_e_drum_raw": pred_state.e_drum_raw,
                "pred_e_drum": pred_state.e_drum,
                "pred_m_burden": pred_state.m_burden,
                "pred_p_dry": pred_state.p_dry,
                "pred_p_mai": pred_state.p_mai,
                "pred_p_dev": pred_state.p_dev,
                "bt_error": pred_state.bt - actual_bt_i,
                "et_error": (pred_state.et - actual_et_i) if not teacher_force_et else float("nan"),
                "ror_error": pred_state.ror - actual_ror_i,
            }
        )

    metrics = ReplayMetrics(
        n_steps=len(rows),
        bt_rmse=_rmse(actual_bt, pred_bt),
        et_rmse=_rmse(actual_et, pred_et) if not teacher_force_et else float("nan"),
        ror_rmse=_rmse(actual_ror, pred_ror),
        bt_mae=_mae(actual_bt, pred_bt),
        et_mae=_mae(actual_et, pred_et) if not teacher_force_et else float("nan"),
        ror_mae=_mae(actual_ror, pred_ror),
        terminal_bt_error=(pred_bt[-1] - actual_bt[-1]) if pred_bt else float("nan"),
        terminal_et_error=(pred_et[-1] - actual_et[-1]) if (pred_et and not teacher_force_et) else float("nan"),
        terminal_ror_error=(pred_ror[-1] - actual_ror[-1]) if pred_ror else float("nan"),
    )

    return ReplayResult(rows=rows, metrics=metrics)


def summarize_replay_metrics(result: ReplayResult) -> dict:
    m = result.metrics
    return {
        "n_steps": m.n_steps,
        "bt_rmse": m.bt_rmse,
        "et_rmse": m.et_rmse,
        "ror_rmse": m.ror_rmse,
        "bt_mae": m.bt_mae,
        "et_mae": m.et_mae,
        "ror_mae": m.ror_mae,
        "terminal_bt_error": m.terminal_bt_error,
        "terminal_et_error": m.terminal_et_error,
        "terminal_ror_error": m.terminal_ror_error,
    }


def replay_roast_from_parquet(
    parquet_path: str,
    simulator: CalibratedRoasterSimulator,
    roast_id: Optional[str] = None,
    teacher_force_et: bool = True,
    teacher_force_ror: bool = True,
    teacher_force_phase: bool = True,
    warmup_rows: int = 120,
    use_estimator: bool = False,
) -> ReplayResult:
    df = pd.read_parquet(parquet_path)
    return replay_roast_dataframe(
        df=df,
        simulator=simulator,
        roast_id=roast_id,
        teacher_force_et=teacher_force_et,
        teacher_force_ror=teacher_force_ror,
        teacher_force_phase=teacher_force_phase,
        warmup_rows=warmup_rows,
        use_estimator=use_estimator,
    )


