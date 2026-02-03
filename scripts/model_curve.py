import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load trained pipeline and data
from joblib import dump

# Reload model and data (retrain quickly to keep state simple)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv(
    '../data/sales_pricing_model_ready.csv'
)

feature_cols = [
    "PRODUCTLINE",
    "QUANTITYORDERED",
    "MSRP",
    "PRICEEACH",
    "DISCOUNT_PCT_CLIPPED",
    "MONTH_ID",
    "YEAR_ID",
]
traget_col = "IS_CLOSED"
X = df[feature_cols]
y = df[traget_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE']),
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', GradientBoostingClassifier(random_state=42)),
])

# Train model
model.fit(X_train, y_train)
# Save trained model
dump(model, '../models/sales_closing_model.joblib')

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
# Feature importance
classifier = model.named_steps['classifier']
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('../models/feature_importances.png')
plt.show()

# Save test set for future evaluation
test_set = X_test.copy()
test_set[traget_col] = y_test
test_set.to_csv('../data/sales_model_test_set.csv', index=False)
print("Saved test set to ../data/sales_model_test_set.csv")

print("Model training and evaluation complete.")

# ---- Discount sweep for one product line ----
product_line = "Classic Cars"
subset = df[df["PRODUCTLINE"] == product_line]

context = {
    "PRODUCTLINE": product_line,
    "QUANTITYORDERED": int(subset["QUANTITYORDERED"].median()),
    "MSRP": float(subset["MSRP"].median()),
    "PRICEEACH": float(subset["PRICEEACH"].median()),
    "MONTH_ID": int(subset["MONTH_ID"].mode()[0]),
    "YEAR_ID": int(subset["YEAR_ID"].mode()[0]),
}

discounts = np.arrange(0, 41, 1.0)  # 0% to 40% discount
results = []

for discount in discounts:
    results = context.copy()
    results["DISCOUNT_PCT_CLIPPED"] = discount
    results.append(discount)

sweep_df = pd.DataFrame(results)
predictions = model.predict_proba(sweep_df)[:, 1]

unit_revenue = context["PRICEEACH"] * (context["QUANTITYORDERED"])
expected_profits = predictions * unit_revenue

plt.figure(figsize=(10, 6))
plt.plot(discounts, expected_profits, marker='o')
plt.title(f"Expected Profit vs Discount Percentage for {product_line}")
plt.xlabel("Discount Percentage (%)")
plt.ylabel("Expected Profit")
plt.grid()

plt.savefig(f'../models/discount_sweep_{product_line.replace(" ", "_")}.png')
plt.show()

print("Discount sweep analysis complete.")