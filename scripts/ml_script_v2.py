import pandas as pd
import numpy as np

INPUT_PATH = "../data/sales_data_sample_cleaned.csv"
OUTPUT_PATH = "../data/sales_pricing_model_ready_v2.csv"

df = pd.read_csv(
    INPUT_PATH,
    dtype={"PHONE": "string"},
    na_values=["NA", "NaN", ""]
)

df.columns = [c.strip().upper() for c in df.columns]
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

for col in ["STATUS", "PRODUCTLINE", "PRODUCTCODE", "COUNTRY", "STATE", "CITY"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# Discount percent: positive = discount, negative = markup
df["DISCOUNT_PERCENTAGE"] = ((df["MSRP"] - df["PRICEEACH"]) / df["MSRP"]) * 100.0

# Closed label (you chose shipped/resolved)
df["IS_CLOSED"] = df["STATUS"].str.lower().isin(["shipped", "resolved"]).astype(int)

# Outliers + clipped decision variable
df["DISCOUNT_OUTLIER_FLAG"] = ((df["DISCOUNT_PERCENTAGE"] < -10) | (df["DISCOUNT_PERCENTAGE"] > 50)).astype(int)
df["DISCOUNT_PCT_CLIPPED"] = df["DISCOUNT_PERCENTAGE"].clip(lower=-10, upper=50)

# Core model columns
model_cols = [
    "ORDERDATE", "YEAR_ID", "MONTH_ID", "QTR_ID",
    "PRODUCTLINE", "QUANTITYORDERED",
    "MSRP", "PRICEEACH",
    "DISCOUNT_PERCENTAGE", "DISCOUNT_PCT_CLIPPED", "DISCOUNT_OUTLIER_FLAG",
    "STATUS", "IS_CLOSED"
]
df_model = df[model_cols].copy()

# Drop missing essentials
essential = ["PRODUCTLINE", "QUANTITYORDERED", "MSRP", "PRICEEACH", "DISCOUNT_PCT_CLIPPED", "IS_CLOSED", "MONTH_ID", "YEAR_ID", "QTR_ID"]
df_model = df_model.dropna(subset=essential).reset_index(drop=True)

# Types
df_model["PRODUCTLINE"] = df_model["PRODUCTLINE"].astype("category")
df_model["MONTH_ID"] = df_model["MONTH_ID"].astype(int)
df_model["YEAR_ID"] = df_model["YEAR_ID"].astype(int)
df_model["QTR_ID"] = df_model["QTR_ID"].astype(int)

# ---- New: exposure + "order health" proxies ----
df_model["DEAL_SIZE"] = df_model["PRICEEACH"] * df_model["QUANTITYORDERED"]
df_model["LOG_DEAL_SIZE"] = np.log1p(df_model["DEAL_SIZE"])

# Price pressure / pricing health proxy
df_model["PRICE_TO_MSRP_RATIO"] = df_model["PRICEEACH"] / df_model["MSRP"]

# Optional time pressure features
df_model["IS_Q4"] = (df_model["QTR_ID"] == 4).astype(int)
df_model["IS_YEAR_END"] = (df_model["MONTH_ID"] == 12).astype(int)

# Keep buckets for interpretation
q1 = df_model["DEAL_SIZE"].quantile(0.33)
q3 = df_model["DEAL_SIZE"].quantile(0.66)
df_model["DEAL_SIZE_BUCKETS"] = pd.cut(
    df_model["DEAL_SIZE"],
    bins=[-float("inf"), q1, q3, float("inf")],
    labels=["Small", "Medium", "Large"],
    include_lowest=True
).astype("category")

# Sanitize status columns by risk levels
def map_status_to_class(status: str) -> int:
    s = str(status).strip().lower().replace("-", " ")
    if s in {"shipped", "resolved"}:
        return 2  # WON
    if s in {"on hold", "in progress"}:
        return 1  # PENDING
    if s in {"cancelled", "disputed"}:
        return 0  # LOST
    return -1

df["STATUS_CLASS"] = df["STATUS"].apply(map_status_to_class)
df = df[df["STATUS_CLASS"] >= 0].copy()
df_model["STATUS_CLASS"] = df["STATUS_CLASS"]


# Sanity: remove impossible MSRP or PRICE
df_model = df_model[(df_model["MSRP"] > 0) & (df_model["PRICEEACH"] > 0)]

df_model.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(df_model))
print("Closed rate:", df_model["IS_CLOSED"].mean())
print("Outlier discounts:", int(df_model["DISCOUNT_OUTLIER_FLAG"].sum()))
print("PRICE_TO_MSRP_RATIO range:", (df_model["PRICE_TO_MSRP_RATIO"].min(), df_model["PRICE_TO_MSRP_RATIO"].max()))
