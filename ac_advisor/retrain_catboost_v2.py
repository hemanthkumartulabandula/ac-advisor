"""
Train CatBoost v2 using top-N features from SHAP. Saves:
  - models/catboost_v2.cbm
  - models/feature_columns_v2.json
  - updates models/model_registry.json with metrics
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

DATA = Path("data/EnergyPredictionDataset_ReadyForModel.csv")
TARGET = "ac_power_proxy"
IMP = Path("models/shap_importance.csv")
REG = Path("models/model_registry.json")

TOP_N = 25  # keep top 25 features

def main():
    df = pd.read_csv(DATA)
    if not IMP.exists():
        raise SystemExit("Run shap_analysis.py first to produce shap_importance.csv")
    imp = pd.read_csv(IMP)
    MANDATORY = {"voltage_drop", "rpm", "bat_voltage"}
    keep = sorted(set(imp["feature"].head(TOP_N)) | MANDATORY)

    X = df.reindex(columns=keep, fill_value=0).select_dtypes(include=[np.number]).copy()
    y = df[TARGET].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        depth=8, learning_rate=0.06, loss_function="RMSE", verbose=0, random_state=42
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae, rmse, r2 = mean_absolute_error(yte, pred), sqrt(((yte - pred)**2).mean()), r2_score(yte, pred)
    print(f"CatBoost v2 → MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/catboost_v2.cbm")
    Path("models/feature_columns_v2.json").write_text(json.dumps(keep, indent=2))
    # update registry
    reg = json.loads(REG.read_text()) if REG.exists() else {}
    reg["catboost_v2"] = {"mae": mae, "rmse": rmse, "r2": r2, "features": len(keep)}
    REG.write_text(json.dumps(reg, indent=2))
    print("Saved models/catboost_v2.cbm & feature_columns_v2.json and updated registry.")

if __name__ == "__main__":
    main()
