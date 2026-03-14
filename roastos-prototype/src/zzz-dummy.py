import pandas as pd

df = pd.read_parquet("c:/Projects/roastos/roastos-prototype/data/processed/calibration_dataset.parquet")

print(df[["sweetness","acidity","overall"]].describe())

print(df["phase"].value_counts())


print(df.head())
print("Columns:", df.columns)



#Fast way to see valid roast IDs

#python -c "import pandas as pd; df=pd.read_parquet('data/processed/calibration_dataset.parquet'); print(sorted(df['roast_id'].dropna().astype(str).unique().tolist()))"

#If that prints too many, use:
#python -c "import pandas as pd; df=pd.read_parquet('data/processed/calibration_dataset.parquet'); ids=sorted(df['roast_id'].dropna().astype(str).unique().tolist()); print(ids[:50]); print('count=', len(ids))"

#If you want only IDs similar to PR-0176
#python -c "import pandas as pd; df=pd.read_parquet('data/processed/calibration_dataset.parquet'); ids=sorted(df['roast_id'].dropna().astype(str).unique().tolist()); print([x for x in ids if '017' in x])"

"""
Best practical approach now

For benchmarking several roasts, only use roasts with enough valid rows.

You can list the roasts with usable replay length like this:

python -c "import pandas as pd; df=pd.read_parquet('data/processed/calibration_dataset.parquet'); cols=['time_s','bt_c','et_c','ror','gas','pressure','drum_speed','phase']; tmp=df.dropna(subset=cols).groupby(df['roast_id'].astype(str)).size().sort_values(ascending=False); print(tmp.to_string())"

That will show how many valid rows each roast has after cleaning.

Then pick roasts with, say, at least 50 rows.
"""