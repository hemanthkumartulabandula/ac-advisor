"""
Train P10 and P90 quantile CatBoost models for prediction intervals.
Outputs:
  - models/cb_q10.cbm
  - models/cb_q90.cbm
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

DATA = Path("data/EnergyPredictionDataset_ReadyForModel.csv")
TARGET = "ac_power_proxy"

# use v2 features if present, else v1
F1 = Path("models/feature_columns_v2.json")
F0 = Path("models/feature_columns.json")
FEATS = F1 if F1.exists() else F0

def train_q(alpha, Xtr, ytr):
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", depth=8, learning_rate=0.06, verbose=0, random_state=42)
    m.fit(Xtr, ytr)
    return m

def main():
    features = json.loads(FEATS.read_text())
    df = pd.read_csv(DATA)
    X = df.reindex(columns=features, fill_value=0).select_dtypes(include=[np.number]).copy()
    y = df[TARGET].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    q10 = train_q(0.10, Xtr, ytr); q10.save_model("models/cb_q10.cbm")
    q90 = train_q(0.90, Xtr, ytr); q90.save_model("models/cb_q90.cbm")
    print("Saved models/cb_q10.cbm and models/cb_q90.cbm")

if __name__ == "__main__":
    main()
