import pandas as pd
import numpy as np

INPUT_PATH = "../data/sales_data_sample_cleaned.csv"
OUTPUT_PATH = "../data/sales_pricing_model_ready.csv"

# 1) Load with safe NA handling and stable dtypes
df = pd.read_csv(
    INPUT_PATH,
    dtype={"PHONE": "string"},
    na_values=["NA", "NaN", ""]
)

# 2) Normalize column names
df.columns = [c.strip().upper() for c in df.columns]

# 3) Parse dates
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

# 4) Clean categorical strings (trim whitespace)
for col in ["STATUS", "PRODUCTLINE", "PRODUCTCODE", "COUNTRY", "STATE", "CITY"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# 5) Create discount percentage (since it's missing in this CSV)
#    Positive = discount, Negative = markup
df["DISCOUNT_PERCENTAGE"] = ((df["MSRP"] - df["PRICEEACH"]) / df["MSRP"]) * 100.0

# 6) Create a clean binary outcome: closed vs not closed
df["IS_CLOSED"] = df["STATUS"].str.lower().isin(["shipped", "resolved"]).astype(int)

# 7) Flag extreme discounts (outliers) and create a clipped version for modeling
df["DISCOUNT_OUTLIER_FLAG"] = (df["DISCOUNT_PERCENTAGE"] < -10) | (df["DISCOUNT_PERCENTAGE"] > 50)
df["DISCOUNT_PCT_CLIPPED"] = df["DISCOUNT_PERCENTAGE"].clip(lower=-10, upper=50)

# 8) Select model-relevant columns only (drop identity and noisy fields)
model_cols = [
    "ORDERDATE", "YEAR_ID", "MONTH_ID", "QTR_ID",
    "PRODUCTLINE", "QUANTITYORDERED",
    "MSRP", "PRICEEACH",
    "DISCOUNT_PERCENTAGE", "DISCOUNT_PCT_CLIPPED", "DISCOUNT_OUTLIER_FLAG",
    "STATUS", "IS_CLOSED"
]

df_model = df[model_cols].copy()

# 9) Drop rows missing essential fields
essential = ["PRODUCTLINE", "QUANTITYORDERED", "MSRP", "PRICEEACH", "DISCOUNT_PCT_CLIPPED", "IS_CLOSED", "MONTH_ID", "YEAR_ID"]
df_model = df_model.dropna(subset=essential)

# 10) Fix types
df_model["PRODUCTLINE"] = df_model["PRODUCTLINE"].astype("category")
df_model["MONTH_ID"] = df_model["MONTH_ID"].astype(int)
df_model["YEAR_ID"] = df_model["YEAR_ID"].astype(int)
df_model["QTR_ID"] = df_model["QTR_ID"].astype(int)

# 11) Ensure boolean flags are int
df_model["DISCOUNT_OUTLIER_FLAG"] = df_model["DISCOUNT_OUTLIER_FLAG"].astype(int)

#12 ) Reset index
df_model = df_model.reset_index(drop=True)

#13) Create Deal_Size
df_model["DEAL_SIZE"] = df_model["PRICEEACH"] * df_model["QUANTITYORDERED"]

q1 = df_model["DEAL_SIZE"].quantile(0.33)
q3 = df_model["DEAL_SIZE"].quantile(0.66)

df_model["DEAL_SIZE_BUCKETS"] = pd.cut(
    df_model["DEAL_SIZE"],
    bins=[-float("inf"), q1, q3, float("inf")],
    labels=["Small", "Medium", "Large"],
    include_lowest=True
).astype("category")

# 11) Save model-ready data
df_model.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(df_model))
print("Closed rate:", df_model["IS_CLOSED"].mean())
print("Outlier discounts:", df_model["DISCOUNT_OUTLIER_FLAG"].sum())
print(df_model["PRODUCTLINE"].value_counts())
