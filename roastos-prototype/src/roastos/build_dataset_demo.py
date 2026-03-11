from pathlib import Path

from roastos.dataset_builder import build_training_data_from_dir

"""Script to demonstrate loading the dataset and building the ML dataframe, printing out the results."""

def main() -> None:
    full_df, X, y = build_training_data_from_dir(Path("data/mock"))

    print("Full joined dataframe:")
    print(full_df.head())
    print("\nShape:", full_df.shape)

    print("\nX (features):")
    print(X.head())
    print("\nX shape:", X.shape)

    print("\ny (targets):")
    print(y.head())
    print("\ny shape:", y.shape)

    print("\nFeature columns:")
    print(list(X.columns))

    print("\nTarget columns:")
    print(list(y.columns))


if __name__ == "__main__":
    main()


"""
It produces one final row per roast containing:

From roast_sessions

roast metadata

flavour intent

ambient conditions

machine id

operator

style profile

From roast_features

%Dry

%Maillard

%Dev

RoR_FC

V_loss_final

S_struct_final

crash/flick indices

From roast_outcomes

clarity

sweetness

body

bitterness

acidity

aroma

From coffee_lots

origin

process

variety

density

moisture

water activity

altitude

So the final dataframe is exactly what you need for:

F^=g(x,θ) this is the ML model to predict flavour from roast structure and context
where:

𝑥
x = roast structure features
θ = context
F = flavour outputs

ML modeling"""

