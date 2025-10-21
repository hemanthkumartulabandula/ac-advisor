# cataboost.py — run once to export model + feature list
import os, json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Use the SAME CSV the app reads (yours is already here)
file_path = "data/EnergyPredictionDataset_ReadyForModel.csv"
df = pd.read_csv(file_path)

# Target column (change if your CSV uses a different name)
target = "ac_power_proxy"

# Build features just like your training approach
non_numeric_cols = ["timestamp", "trip_id", "weather_desc", "ac_on_label_custom"]
X = df.drop(columns=[target] + non_numeric_cols, errors="ignore")
X = X.select_dtypes(include=[np.number]).copy()
y = df[target].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train CatBoost (quiet)
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train, y_train)

# Quick eval
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print("\n===== CATBOOST EVALUATION =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

# Save artifacts for the app
os.makedirs("models", exist_ok=True)

# 1) CatBoost native model (what the app loads)
cbm_path = "models/catboost_model.cbm"
model.save_model(cbm_path)
print(f"\nSaved model → {cbm_path}")

# 2) Exact feature list used during training
feature_list_path = "models/feature_columns.json"
with open(feature_list_path, "w") as f:
    json.dump(X.columns.tolist(), f, indent=2)
print(f"Saved features → {feature_list_path}")

print("\nDone. Restart the app to use your real model.")
