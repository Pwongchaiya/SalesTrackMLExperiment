import pandas as pd
from utils.normailize_phone_numbers_to_e164 import normalize_phone_to_e164
from utils.geocoding import geocoding

df = pd.read_csv(
    '../data/sales_data_sample.csv', 
    encoding='latin-1'
)

## Drop columns
columns_to_drop = [
    "ADDRESSLINE1",
    "ADDRESSLINE2",
    "POSTALCODE",
    "CONTACTFIRSTNAME",
    "CONTACTLASTNAME",
    "DEALSIZE",
]

df = df.drop(columns=columns_to_drop)

# Fix Glen Waverly typo
cities = {
    "Glen Waverly": "Glen Waverley",
    "Gensve": "Geneva",
    "Aaarhus": "Aarhus",
    "Tsawassen": "Tsawwassen",
}
df["CITY"] = df["CITY"].replace(cities)


# Geocoding locations
geocoding(df)

# Normalize phone numbers to E.164 format
df["PHONE"] = df.apply(
        lambda row: normalize_phone_to_e164(row.get("PHONE"), row.get("COUNTRY"), row.get("CITY")),
        axis=1
    )

# Convert to strict data types
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

numeric_columns = [
    "SALES",
    "PRICEEACH",
    "MSRP",
    "QUANTITYORDERED",
    "LATITUDE",
    "LONGITUDE",
]

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Strip whitespace from text columns
text_columns = [
    "STATUS",
    "COUNTRY",
    "CITY",
    "PRODUCTLINE",
    "PRODUCTCODE",
]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Fill in missing TERRITORY for USA and Canada
for territory, country, city in df[["TERRITORY","COUNTRY","CITY"]].itertuples(index=False):
    if pd.isna(territory) and country in ["USA", "Canada"]:
        df["TERRITORY"] = "NA"

# Drop duplicate rows
df = df.drop_duplicates()

# Analyze missing data
df.isna().mean().sort_values(ascending=False)

df.info()
df.describe(include='all')
df.to_csv("../data/sales_data_sample_cleaned.csv", index=False)

