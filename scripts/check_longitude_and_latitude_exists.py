import pandas as pd

""""Check for missing LONGITUDE and LATITUDE values in the cleaned data CSV."""

df = pd.read_csv(
    '../data/sales_data_sample_cleaned.csv',
    encoding='latin-1'
)

print(df[["LONGITUDE","LATITUDE"]].isna().sum())
for col in ["LONGITUDE","LATITUDE"]:
    missing = df[df[col].isna()]
    if not missing.empty:
        print(f"Rows with missing {col}:")
        print(missing[["CITY","COUNTRY",col]].drop_duplicates())