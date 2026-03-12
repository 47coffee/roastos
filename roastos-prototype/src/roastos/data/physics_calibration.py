from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATASET_PATH = Path("data/processed/calibration_dataset.parquet")
OUTPUT_PATH = Path("artifacts/models/physics_model.json")


def load_dataset():

    if not DATASET_PATH.exists():
        raise RuntimeError("Calibration dataset not found. Run dataset_builder first.")

    df = pd.read_parquet(DATASET_PATH)

    return df

def prepare_training_matrix(df):

    df = df.copy()

    # compute next temperature
    df["bt_next"] = df.groupby("roast_id")["bt_c"].shift(-1)

    # ET-BT heat transfer
    df["et_delta"] = df["et_c"] - df["bt_c"]

    # remove rows with missing critical data
    df = df.dropna(
        subset=[
            "bt_c",
            "bt_next",
            "gas",
            "pressure",
            "ror",
            "et_delta",
        ]
    )

    # target variable
    y = df["bt_next"] - df["bt_c"]

    # feature matrix
    X = df[
        [
            "gas",
            "et_delta",
            "pressure",
            "ror",
        ]
    ]

    return X, y

def fit_physics_model(X, y):

    model = LinearRegression()

    model.fit(X, y)

    coeffs = {
        "alpha_gas": model.coef_[0],
        "beta_et": model.coef_[1],
        "gamma_pressure": model.coef_[2],
        "delta_ror": model.coef_[3],
        "intercept": model.intercept_,
    }

    return model, coeffs


def save_model(coeffs):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(OUTPUT_PATH, "w") as f:
        json.dump(coeffs, f, indent=2)

    print("Physics model saved:")
    print(OUTPUT_PATH)


def main():

    print("Loading dataset...")

    df = load_dataset()

    print("Rows:", len(df))

    X, y = prepare_training_matrix(df)

    print("Training samples:", len(X))

    model, coeffs = fit_physics_model(X, y)

    print("\nEstimated physics coefficients:\n")

    for k, v in coeffs.items():
        print(f"{k:15s} {v: .6f}")

    save_model(coeffs)


if __name__ == "__main__":

    main()