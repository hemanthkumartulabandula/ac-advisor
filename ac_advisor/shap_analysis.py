"""
Run SHAP on CatBoost v1 to get global importances and per-row explanations.
Outputs:
  - models/shap_importance.csv
  - models/shap_sample_row.json (top k contributions for a sample row)
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

DATA = Path("data/EnergyPredictionDataset_ReadyForModel.csv")
MODEL = Path("models/catboost_model.cbm")
FEATS = Path("models/feature_columns.json")
OUT_IMP = Path("models/shap_importance.csv")
OUT_ROW = Path("models/shap_sample_row.json")

SAMPLE_ROW_INDEX = 50   # change in app at runtime if you like

def main():
    model = CatBoostRegressor()
    model.load_model(str(MODEL))
    features = json.loads(FEATS.read_text())
    df = pd.read_csv(DATA)
    X = df.reindex(columns=features, fill_value=0)

    # CatBoost native SHAP
    pool = Pool(X)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    # shap_vals shape = (n_rows, n_features + 1) last col is base value
    phi = shap_vals[:, :-1]
    names = features

    # Global importance = mean(|phi|)
    imp = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(phi).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    OUT_IMP.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(OUT_IMP, index=False)
    print(f"Wrote {OUT_IMP} (top 10)\n", imp.head(10))

    # One-row explanation (for app demo)
    r = min(SAMPLE_ROW_INDEX, len(X)-1)
    row_phi = pd.Series(phi[r], index=names).sort_values(key=np.abs, ascending=False)
    row_out = [{"feature": k, "shap": float(v)} for k, v in row_phi.head(12).items()]
    OUT_ROW.write_text(json.dumps({"row_index": r, "contribs": row_out}, indent=2))
    print(f"Wrote {OUT_ROW}")

if __name__ == "__main__":
    main()
