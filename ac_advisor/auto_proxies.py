from pathlib import Path
import json
import pandas as pd

DEFAULTS = {
    "setpoint": "temp_rpm_interaction",
    "fan": "speed",
    "recirc": "humidity_over_voltage",
}

CANDIDATES = {
    "setpoint": ["temp_rpm_interaction", "ambient_temp", "temperature", "cabin_temp", "outside_air_temp", "oat", "ambient"],
    "fan": ["speed_rollmean_3", "rpm_rollmean_3", "speed", "rpm", "airflow"],
    "recirc": ["humidity_over_voltage", "relative_humidity", "humidity", "dew_point"],
}

def pick_from_names(feature_cols, name_list):
    for n in name_list:
        if n in feature_cols:
            return n
    # substring match fallback
    for n in name_list:
        for f in feature_cols:
            if n in f:
                return f
    return None

def from_shap(feature_cols, shap_csv="models/shap_importance.csv"):
    try:
        imp = pd.read_csv(shap_csv)
    except Exception:
        return []  
    ranked = [f for f in imp["feature"].tolist() if f in feature_cols]
    return ranked

def select_proxies(feature_cols, shap_csv="models/shap_importance.csv"):
    """Return a dict {'setpoint': col, 'fan': col, 'recirc': col} choosing real, influential columns."""
    chosen = {}
    ranked = from_shap(feature_cols, shap_csv)  

    
    def pick_with_rank(key):
        # 1) exact/substring candidates first
        cand = pick_from_names(feature_cols, CANDIDATES[key])
        # 2) if not found, pick highest SHAP feature that "looks related"
        if not cand and ranked:
            keywords = CANDIDATES[key]
            for f in ranked:
                if any(k in f for k in keywords):
                    cand = f
                    break
        # 3) fallback to highest SHAP feature overall
        if not cand and ranked:
            cand = ranked[0]
        # 4) final fallback to default if present in features, else any first feature
        if not cand:
            d = DEFAULTS[key]
            cand = d if d in feature_cols else (feature_cols[0] if feature_cols else d)
        return cand

    chosen["setpoint"] = pick_with_rank("setpoint")
    chosen["fan"]      = pick_with_rank("fan")
    chosen["recirc"]   = pick_with_rank("recirc")
    return chosen
