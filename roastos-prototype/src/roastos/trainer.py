from __future__ import annotations

from pathlib import Path
import json
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from roastos.dataset_builder import build_training_data_from_dir


"""Script to train one model per flavor target and save the models and a summary of results."""
"""For simplicity, we train and evaluate on the same dataset for now.
Later we will replace this with proper train/validation splits."""
"""We also print out feature importances to get insights into which features are most influential for each flavor target."""
"""The models are saved to disk using joblib, and a summary JSON file is created containing the evaluation 
metrics and top features for each target."""

"""Flavor targets we want to predict - these should match the columns in the dataset builder output."""
"""It trains 4 separate models:
clarity_model.joblib
sweetness_model.joblib
body_model.joblib
bitterness_model.joblib"""

"""For the prototype, I recommend Random Forest Regressor first because:
    it works well on tiny / messy tabular datasets
    no scaling needed
    interpretable enough
    robust for v1
Later we can switch to:
    XGBoost
    LightGBM
    multi-output models
    uncertainty-aware models"""

TARGET_COLUMNS = [
    "clarity",
    "sweetness",
    "body",
    "bitterness",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_single_target_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train one regressor for one flavor target.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Evaluate on the same dataset for now.
    Later we will replace this with proper train/validation splits.
    """
    preds = model.predict(X)

    metrics = {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(mean_squared_error(y, preds) ** 0.5),
        "r2": float(r2_score(y, preds)),
    }
    return metrics


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    *,
    top_n: int = 15,
) -> list[dict]:
    """
    Return sorted feature importances.
    """
    importances = model.feature_importances_
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, importances)
    ]
    rows = sorted(rows, key=lambda x: x["importance"], reverse=True)
    return rows[:top_n]


def train_all_targets(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    model_dir: Path,
) -> dict:
    """
    Train one model per target and save each model.
    """
    ensure_dir(model_dir)

    summary: dict = {}

    for target in TARGET_COLUMNS:
        print(f"\n--- Training model for target: {target} ---")

        target_y = y[target]
        model = train_single_target_model(X, target_y)
        metrics = evaluate_model(model, X, target_y)
        importance = get_feature_importance(model, list(X.columns), top_n=15)

        model_path = model_dir / f"{target}_model.joblib"
        joblib.dump(model, model_path)

        summary[target] = {
            "model_path": str(model_path),
            "metrics": metrics,
            "top_features": importance,
        }

        print(f"Saved model to: {model_path}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("Top feature importances:")
        for row in importance[:10]:
            print(f"  {row['feature']}: {row['importance']:.4f}")

    return summary


def save_training_summary(summary: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    data_dir = Path("data/mock")
    model_dir = Path("artifacts/models")
    summary_path = Path("artifacts/training_summary.json")

    print("Loading ML-ready dataset...")
    full_df, X, y = build_training_data_from_dir(
        data_dir=data_dir,
        encode_categories=True,
        target_columns=TARGET_COLUMNS,
    )

    print("\nDataset loaded.")
    print(f"Full dataframe shape: {full_df.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nFeature columns:")
    for c in X.columns:
        print(f"  - {c}")

    print("\nTraining models...")
    summary = train_all_targets(X, y, model_dir=model_dir)

    save_training_summary(summary, summary_path)

    print(f"\nTraining summary saved to: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()