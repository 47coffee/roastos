from pathlib import Path

import pandas as pd

from roastos.dataset_builder import build_training_data_from_dir
from roastos.predictor import FlavorPredictor

"""This script demonstrates how to use the FlavorPredictor to make predictions on new data. 
It builds a sample ML-ready dataframe from the mock dataset, initializes the predictor with the trained 
models, and then performs both single-row and batch predictions, printing the results to the console. 
This serves as a simple example of how to use the predictor in practice after training the models."""

def main() -> None:
    # Build a sample ML-ready dataframe from the mock dataset
    _, X, _ = build_training_data_from_dir(
        data_dir=Path("data/mock"),
        encode_categories=True,
        target_columns=["clarity", "sweetness", "body", "bitterness"],
    )

    predictor = FlavorPredictor(model_dir=Path("artifacts/models"))

    print("Required features:")
    for f in predictor.get_required_features():
        print(f"  - {f}")

    # Predict one row
    print("\nSingle-row prediction:")
    pred = predictor.predict_row(X.iloc[0])
    print(pred)
    print(pred.to_dict())

    # Predict many rows
    print("\nBatch prediction:")
    pred_df = predictor.predict_dataframe(X.head(3))
    print(pred_df)


if __name__ == "__main__":
    main()