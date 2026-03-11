from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

"""This module defines the FlavorPredictor class, which loads trained machine learning models for predicting 
coffee flavor profiles based on input features. The predictor can handle both single-row and batch predictions, 
ensuring that the required features are present in the input data. The predicted flavor profiles include clarity, 
sweetness, body, and bitterness, which are returned  as a FlavorPrediction dataclass for easy access and 
conversion to a dictionary format. This class serves as the core component for making flavor predictions in 
the RoastOS system after the models have been trained and saved to disk."""     

TARGET_COLUMNS = [
    "clarity",
    "sweetness",
    "body",
    "bitterness",
]


@dataclass
class FlavorPrediction:
    clarity: float
    sweetness: float
    body: float
    bitterness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "clarity": self.clarity,
            "sweetness": self.sweetness,
            "body": self.body,
            "bitterness": self.bitterness,
        }


class FlavorPredictor:
    """
    Loads one trained model per target and predicts flavor outputs
    from ML-ready feature inputs.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.models: dict[str, Any] = {}
        self.feature_names: list[str] | None = None
        self._load_models()

    def _load_models(self) -> None:
        for target in TARGET_COLUMNS:
            model_path = self.model_dir / f"{target}_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")

            model = joblib.load(model_path)
            self.models[target] = model

            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
                if self.feature_names is None:
                    self.feature_names = names
                elif self.feature_names != names:
                    raise ValueError(
                        f"Feature mismatch across models. "
                        f"Expected {self.feature_names}, got {names} in {target}"
                    )

        if self.feature_names is None:
            raise ValueError(
                "Could not determine feature_names_in_ from trained models. "
                "Make sure models were trained with a pandas DataFrame."
            )

    def get_required_features(self) -> list[str]:
        return list(self.feature_names or [])

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.feature_names is not None

        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns for prediction: {missing}"
            )

        return X[self.feature_names].copy()

    def predict_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        X_prepared = self._prepare_dataframe(X)

        preds: dict[str, Any] = {}
        for target, model in self.models.items():
            preds[target] = model.predict(X_prepared)

        return pd.DataFrame(preds, index=X.index)

    def predict_row(self, row: dict[str, Any] | pd.Series) -> FlavorPrediction:
        row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        X = pd.DataFrame([row_dict])
        pred_df = self.predict_dataframe(X)
        pred = pred_df.iloc[0]

        return FlavorPrediction(
            clarity=float(pred["clarity"]),
            sweetness=float(pred["sweetness"]),
            body=float(pred["body"]),
            bitterness=float(pred["bitterness"]),
        )