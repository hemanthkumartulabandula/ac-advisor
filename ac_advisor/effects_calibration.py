"""
Calibrates effect multipliers for setpoint Δ, fan level, and recirc using PDP-like sweeps.
Fix: align columns to model's feature list and coerce datetimes/strings to numeric before predict.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

MODEL_PATH = Path("models/catboost_model.cbm")
FEATS_PATH = Path("models/feature_columns.json")
DATA_PATH  = Path("data/EnergyPredictionDataset_ReadyForModel.csv")
OUT_PATH   = Path("ac_advisor/effects.json")

# --- knobs ---
N_BASE_ROWS = 2000
SETPOINT_SWEEP = np.array([-10,-6,-3,-1,0,1,3,6,10])
FAN_SWEEP       = np.array([-2,-1,0,1,2])
RECIRC_STATES   = [0,1]
TEMP_RPM_FEATURE = "temp_rpm_interaction"
AIRFLOW_PROXY    = "speed"
HUM_OVER_VOLT    = "humidity_over_voltage"


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce non-numeric cols (including datetimes) to numeric; fill NaNs with 0."""
    df = df.copy()
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            continue
        # try datetime
        try:
            dt = pd.to_datetime(df[c], errors="raise", utc=True)
            # convert to seconds since epoch (float)
            df[c] = dt.view("int64") / 1e9
            continue
        except Exception:
            pass
        # fallback: numeric coercion
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0.0)

def _align_for_model(X: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Select and order exactly the model's feature columns, coerce to numeric."""
    Xf = X.reindex(columns=feature_cols, fill_value=0)
    return _coerce_numeric(Xf)

def load():
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    with open(FEATS_PATH) as f:
        feature_cols = json.load(f)
    X = pd.read_csv(DATA_PATH)
    return model, feature_cols, X

def pdp_effect(model, feature_cols, X, proxy_col, change_fn, sweep):
    """Return slope (per unit of control) by regressing Δprediction vs control."""
    # Sample rows once from the raw frame, then align/clean
    base_raw = X.sample(min(N_BASE_ROWS, len(X)), random_state=42).copy()
    base = _align_for_model(base_raw, feature_cols)
    base_pred = model.predict(base)

    deltas, controls = [], []
    for c in sweep:
        mod_raw = base_raw.copy()
        if proxy_col in mod_raw.columns:
            mod_raw[proxy_col] = change_fn(mod_raw[proxy_col].values, c)
        mod = _align_for_model(mod_raw, feature_cols)
        pred_c = model.predict(mod)
        deltas.append(base_pred - pred_c)  # positive = saving
        controls.append(c)

    deltas = np.vstack(deltas).T
    med = np.median(deltas, axis=0)
    slope = np.polyfit(controls, med, 1)[0]
    return float(slope)

def main():
    model, feature_cols, X = load()

    # Ensure proxies exist even if missing in CSV
    for col in [TEMP_RPM_FEATURE, AIRFLOW_PROXY, HUM_OVER_VOLT]:
        if col not in X:
            X[col] = 1.0

    # Define how controls perturb proxy columns (same directionality as app)
    def _clean_array(v):
    # Accepts Series/ndarray; returns ndarray with NaNs->0
        arr = pd.Series(v) if not isinstance(v, pd.Series) else v
        arr = pd.to_numeric(arr, errors="coerce")
        return np.nan_to_num(arr.to_numpy(), nan=0.0)

    def setpoint_change(v, dT):  # +dT reduces cooling-load proxy
        arr = _clean_array(v)
        return arr * (1.0 - 0.02 * dT)

    def fan_change(v, dfan):     # +fan increases airflow proxy
        arr = _clean_array(v)
        return arr * (1.0 + 0.01 * dfan)

    def recirc_apply(df, on):    # ON reduces humidity proxy
        df = df.copy()
        if HUM_OVER_VOLT in df.columns:
            arr = _clean_array(df[HUM_OVER_VOLT])
            df[HUM_OVER_VOLT] = arr * (1.0 - 0.03 * on)
        return df

    # --- PDP-like slopes ---
    setpoint_slope = pdp_effect(model, feature_cols, X, TEMP_RPM_FEATURE, setpoint_change, SETPOINT_SWEEP)
    fan_slope      = pdp_effect(model, feature_cols, X, AIRFLOW_PROXY,    fan_change,      FAN_SWEEP)

    # --- Binary treatment effect for recirc ---
    base_raw = X.sample(min(N_BASE_ROWS, len(X)), random_state=43).copy()
    base = _align_for_model(base_raw, feature_cols)
    base_pred = model.predict(base)

    on_raw = recirc_apply(base_raw, 1)
    on = _align_for_model(on_raw, feature_cols)
    on_pred = model.predict(on)
    recirc_gain = float(np.median(base_pred - on_pred))

    # --- Per-vehicle scaling (best-effort) ---
    vehicle_type_col = None
    for c in X.columns:
        if c.lower() in ("vehicle_type","veh_type","type"):
            vehicle_type_col = c
            break

    per_vehicle = {"default": 1.00, "ICE": 1.00, "HEV": 1.05, "PHEV": 1.10, "EV": 1.20}
    if vehicle_type_col:
        groups = {}
        for g, gdf in X.groupby(vehicle_type_col):
            gdf = gdf.copy()
            if TEMP_RPM_FEATURE not in gdf:
                gdf[TEMP_RPM_FEATURE] = 1.0
            base_g = _align_for_model(gdf.sample(min(2000, len(gdf)), random_state=45), feature_cols)
            gpred = model.predict(base_g)

            mod_g_raw = gdf.sample(min(2000, len(gdf)), random_state=46).copy()
            mod_g_raw[TEMP_RPM_FEATURE] = pd.to_numeric(mod_g_raw[TEMP_RPM_FEATURE], errors="coerce").fillna(0).values * (1 - 0.02*1.0)
            mod_g = _align_for_model(mod_g_raw, feature_cols)
            gpred2 = model.predict(mod_g)
            groups[str(g)] = float(np.median(gpred - gpred2))
        base = groups.get("ICE", np.median(list(groups.values())) if groups else 1.0) or 1.0
        per_vehicle = {k: round(max(0.7, v/base), 2) for k,v in groups.items() if np.isfinite(v)}
        per_vehicle.setdefault("default", 1.0)
        for k in ["ICE","HEV","PHEV","EV"]:
            per_vehicle.setdefault(k, 1.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "setpoint_slope_W_per_deg": round(setpoint_slope, 4),
            "fan_slope_W_per_level": round(fan_slope, 4),
            "recirc_gain_W": round(recirc_gain, 2),
            "per_vehicle_scale": per_vehicle
        }, f, indent=2)
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
