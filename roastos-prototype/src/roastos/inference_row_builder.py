from __future__ import annotations

from typing import Any

import pandas as pd

"""This module defines the build_inference_row function, which constructs a single ML-ready inference row based on the 
extracted roast features, session context, and coffee context. The function ensures that the resulting row matches exactly 
the feature schema used during model training, including handling of numeric features, categorical one-hot encoding, 
and optional timestamp fields. This allows the constructed row to be directly fed into the FlavorPredictor for 
making flavor predictions based on the current roast and context information."""

def _safe_value(source: dict[str, Any], key: str, default: Any = 0) -> Any:
    return source[key] if key in source else default


def build_inference_row(
    *,
    roast_features: dict[str, Any],
    session_context: dict[str, Any],
    coffee_context: dict[str, Any],
    required_feature_names: list[str],
) -> dict[str, Any]:
    """
    Build a single ML-ready inference row matching exactly the trained feature schema.

    Inputs:
    - roast_features: output from extract_features()
    - session_context: batch/intent/style/machine/session info
    - coffee_context: density/moisture/process/origin/etc.
    - required_feature_names: predictor.get_required_features()
    """
    row: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Base numeric roast structure features
    # ------------------------------------------------------------------
    feature_aliases = {
        "pct_dry": roast_features.get("dry", 0.0),
        "pct_maillard": roast_features.get("maillard", 0.0),
        "pct_dev": roast_features.get("dev", 0.0),
        "ror_fc": roast_features.get("ror_fc", 0.0),
        "v_loss_final": roast_features.get("volatile_loss", 0.0),
        "s_struct_final": roast_features.get("structure", 0.0),
        "crash_index": roast_features.get("crash_index", 0.0),
        "flick_index": roast_features.get("flick_index", 0.0),
        "time_to_yellow_s": roast_features.get("time_to_yellow_s", 0),
        "time_to_fc_s": roast_features.get("time_to_fc_s", 0),
        "dev_time_s": roast_features.get("dev_time_s", 0),
        "delta_bt_fc_to_drop_c": roast_features.get("delta_bt_fc_to_drop_c", 0.0),
    }

    # ------------------------------------------------------------------
    # 2. Session context numeric fields
    # ------------------------------------------------------------------
    session_numeric = {
        "batch_size_kg": _safe_value(session_context, "batch_size_kg", 0.0),
        "charge_temp_c": _safe_value(session_context, "charge_temp_c", 0.0),
        "drop_temp_c": _safe_value(session_context, "drop_temp_c", 0.0),
        "duration_s": _safe_value(session_context, "duration_s", 0),
        "ambient_temp_c": _safe_value(session_context, "ambient_temp_c", 0.0),
        "ambient_rh_pct": _safe_value(session_context, "ambient_rh_pct", 0.0),
        "intent_clarity": _safe_value(session_context, "intent_clarity", 0.0),
        "intent_sweetness": _safe_value(session_context, "intent_sweetness", 0.0),
        "intent_body": _safe_value(session_context, "intent_body", 0.0),
        "intent_bitterness": _safe_value(session_context, "intent_bitterness", 0.0),
    }

    # ------------------------------------------------------------------
    # 3. Coffee context numeric fields
    # ------------------------------------------------------------------
    coffee_numeric = {
        "density": _safe_value(coffee_context, "density", 0.0),
        "moisture": _safe_value(coffee_context, "moisture", 0.0),
        "water_activity": _safe_value(coffee_context, "water_activity", 0.0),
        "screen_size": _safe_value(coffee_context, "screen_size", 0.0),
        "altitude_m": _safe_value(coffee_context, "altitude_m", 0),
    }

    # ------------------------------------------------------------------
    # 4. Optional timestamp if model was trained with it
    # ------------------------------------------------------------------
    if "timestamp_start" in required_feature_names:
        timestamp_value = session_context.get("timestamp_start", 0)
        if timestamp_value:
            try:
                row["timestamp_start"] = int(pd.Timestamp(timestamp_value).timestamp())
            except Exception:
                row["timestamp_start"] = 0
        else:
            row["timestamp_start"] = 0

    # ------------------------------------------------------------------
    # 5. Put numeric values into row
    # ------------------------------------------------------------------
    row.update(feature_aliases)
    row.update(session_numeric)
    row.update(coffee_numeric)

    # ------------------------------------------------------------------
    # 6. Categorical one-hot reconstruction
    # Match exactly the names used by pandas.get_dummies during training
    # ------------------------------------------------------------------
    categorical_values = {
        "machine_id": session_context.get("machine_id"),
        "coffee_id": session_context.get("coffee_id"),
        "operator_id": session_context.get("operator_id"),
        "style_profile": session_context.get("style_profile"),
        "brew_method": session_context.get("brew_method"),
        "origin": coffee_context.get("origin"),
        "process": coffee_context.get("process"),
        "variety": coffee_context.get("variety"),
    }

    for feature_name in required_feature_names:
        if feature_name in row:
            continue

        matched = False
        for prefix, value in categorical_values.items():
            expected_prefix = f"{prefix}_"
            if feature_name.startswith(expected_prefix):
                category_value = feature_name[len(expected_prefix) :]
                row[feature_name] = 1 if str(value) == category_value else 0
                matched = True
                break

        if not matched and feature_name not in row:
            row[feature_name] = 0

    # ------------------------------------------------------------------
    # 7. Final exact ordering cleanup
    # ------------------------------------------------------------------
    ordered_row = {feature: row.get(feature, 0) for feature in required_feature_names}
    return ordered_row