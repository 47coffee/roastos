import pandas as pd

df = pd.read_parquet("c:/Projects/roastos/roastos-prototype/data/processed/calibration_dataset.parquet")

print(df[["sweetness","acidity","overall"]].describe())

print(df["phase"].value_counts())


print(df.head())
print("Columns:", df.columns)