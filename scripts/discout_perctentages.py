import pandas as pd

df = pd.read_csv(
    '../data/sales_data_sample_cleaned.csv',
)

# Calculate discount percentages
df['DISCOUNT_PERCENTAGE'] = ((df['MSRP'] - df['PRICEEACH']) / df['MSRP']) * 100
df['DISCOUNT_PERCENTAGE'] = df['DISCOUNT_PERCENTAGE'].round(2)
df['DISCOUNT_PERCENTAGE'] = df['DISCOUNT_PERCENTAGE'].fillna(0)
df.loc[df['MSRP'] == 0, 'DISCOUNT_PERCENTAGE'] = 0

df['IS_CLOSED'] = df['STATUS'].str.lower().isin(['shipped', 'resolved']).astype(int)