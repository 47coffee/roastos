from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data_loader import RoastOSDataset, load_full_dataset


DEFAULT_FEATURE_COLUMNS = [
    # Roast structure
    "pct_dry",
    "pct_maillard",
    "pct_dev",
    "ror_fc",
    "v_loss_final",
    "s_struct_final",
    "crash_index",
    "flick_index",
    "time_to_yellow_s",
    "time_to_fc_s",
    "dev_time_s",
    "delta_bt_fc_to_drop_c",
    # Roast session / operating context
    "batch_size_kg",
    "charge_temp_c",
    "drop_temp_c",
    "duration_s",
    "ambient_temp_c",
    "ambient_rh_pct",
    "intent_clarity",
    "intent_sweetness",
    "intent_body",
    "intent_bitterness",
    # Coffee context
    "density",
    "moisture",
    "water_activity",
    "screen_size",
    "altitude_m",
]

DEFAULT_TARGET_COLUMNS = [
    "clarity",
    "sweetness",
    "body",
    "bitterness",
    "acidity",
    "aroma",
]


def _to_dataframe(records: list) -> pd.DataFrame:
    return pd.DataFrame([model.model_dump() for model in records])


def build_training_dataframe(dataset: RoastOSDataset) -> pd.DataFrame:
    """
    Join sessions + features + outcomes + coffee lots into one ML-ready dataframe.
    """
    sessions_df = _to_dataframe(dataset.sessions)
    features_df = _to_dataframe(dataset.features)
    outcomes_df = _to_dataframe(dataset.outcomes)
    coffee_df = _to_dataframe(dataset.coffee_lots)

    # Join roast_features to roast_sessions
    df = sessions_df.merge(
        features_df,
        on="roast_id",
        how="inner",
        validate="one_to_one",
    )

    # Join roast_outcomes
    df = df.merge(
        outcomes_df,
        on="roast_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )

    # Join coffee_lots via coffee_id from roast_sessions
    df = df.merge(
        coffee_df,
        on="coffee_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_coffee"),
    )

    # Keep only columns useful for ML / tracking
    # but do NOT drop identifiers yet; they can help for debugging
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical fields so the dataframe becomes ML-ready
    for scikit-learn / xgboost style models.
    """
    categorical_cols = [
        "machine_id",
        "coffee_id",
        "operator_id",
        "style_profile",
        "brew_method",
        "origin",
        "process",
        "variety",
    ]

    existing_categoricals = [c for c in categorical_cols if c in df.columns]

    encoded_df = pd.get_dummies(
        df,
        columns=existing_categoricals,
        drop_first=False,
        dtype=int,
    )

    # Convert datetime to numeric if present
    if "timestamp_start" in encoded_df.columns:
        encoded_df["timestamp_start"] = pd.to_datetime(
            encoded_df["timestamp_start"]
        ).astype("int64") // 10**9

    return encoded_df


def build_ml_dataframe(
    dataset: RoastOSDataset,
    *,
    encode_categories: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline:
    - join tables
    - optionally one-hot encode categorical columns
    """
    df = build_training_dataframe(dataset)

    if encode_categories:
        df = encode_categoricals(df)

    return df


def select_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ML dataframe into X (features) and y (targets).
    If feature_columns is None, use DEFAULT_FEATURE_COLUMNS plus encoded categorical columns.
    """
    target_columns = list(target_columns or DEFAULT_TARGET_COLUMNS)

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    if feature_columns is None:
        base_features = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]

        excluded = set(
            [
                "roast_id",
                "notes",
                "notes_outcome",
                "overall_score",
                "panel_size",
                "rest_days",
            ]
            + target_columns
        )

        encoded_extra_features = [
            c
            for c in df.columns
            if c not in excluded
            and c not in base_features
            and (
                c.startswith("machine_id_")
                or c.startswith("coffee_id_")
                or c.startswith("operator_id_")
                or c.startswith("style_profile_")
                or c.startswith("brew_method_")
                or c.startswith("origin_")
                or c.startswith("process_")
                or c.startswith("variety_")
                or c == "timestamp_start"
            )
        ]

        feature_columns = base_features + encoded_extra_features
    else:
        feature_columns = list(feature_columns)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    return X, y


def build_training_data_from_dir(
    data_dir: str | Path,
    *,
    encode_categories: bool = True,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
    - load validated dataset from directory
    - build ML dataframe
    - return (full_df, X, y)
    """
    dataset = load_full_dataset(data_dir)
    full_df = build_ml_dataframe(dataset, encode_categories=encode_categories)
    X, y = select_xy(
        full_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    return full_df, X, y