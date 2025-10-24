# ac_advisor/coach2.py
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd

LOG_PATH = Path("logs/interactions.jsonl")

# ---- Bands for your dataset (tightened): Heating ≤9°C, Cooling ≥15°C ----
def band_from_temp(ambient_c: float) -> str:
    if ambient_c <= 9:
        return "cold"
    if ambient_c >= 15:
        return "hot"
    return "mild"

def speed_bucket(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "na"
    if v < 10: return "0-10"
    if v < 30: return "10-30"
    if v < 60: return "30-60"
    return "60+"

def _read_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(LOG_PATH, lines=True)
    except Exception:
        return pd.DataFrame()

    # Ensure columns exist
    needed = ["ambient_c", "saving_W", "setpoint_delta", "fan_delta", "recirc_on"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    # Derive band and speed bucket if missing
    if "band" not in df.columns:
        df["band"] = df["ambient_c"].apply(band_from_temp)
    if "speed" in df.columns:
        df["speed_bucket"] = df["speed"].apply(speed_bucket)
    else:
        df["speed_bucket"] = "na"
    return df

def _mean(series: pd.Series):
    s = series.dropna()
    return float(s.mean()) if len(s) else None

def _is_nan(x) -> bool:
    try:
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def confidence(n: int) -> str:
    # ●○○ low, ●●○ medium, ●●● high
    return "●○○" if n < 20 else ("●●○" if n < 100 else "●●●")

def nearest_context_advice(ambient_c: float, speed: float | None) -> dict:
    """
    Returns ranked actions for current context using your history:
    band (cold/mild/hot) + speed bucket.
    """
    df = _read_logs()
    band = band_from_temp(ambient_c)
    sb = speed_bucket(speed) if speed is not None else "na"

    if df.empty:
        return {"explanation": "No history yet.", "n": 0, "actions": []}

    ctx = df[df["band"] == band]
    if "speed_bucket" in df.columns and sb != "na":
        ctx = ctx[ctx["speed_bucket"] == sb]

    n = int(len(ctx))
    if n == 0:
        return {"explanation": f"No close matches (band={band}, speed={sb}).", "n": 0, "actions": []}

    # Observed marginal savings by control (+ means saves W on average)
    sp_up      = _mean(ctx.loc[ctx["setpoint_delta"] > 0, "saving_W"])     # warmer
    sp_down    = _mean(ctx.loc[ctx["setpoint_delta"] < 0, "saving_W"])     # cooler
    fan_down   = _mean(ctx.loc[ctx["fan_delta"]      < 0, "saving_W"])     # lower fan
    recirc_on  = _mean(ctx.loc[ctx["recirc_on"]   == True, "saving_W"])    # recirc ON

    # Also keep sample counts for transparency
    n_sp_up    = int((ctx["setpoint_delta"] > 0).sum())
    n_sp_down  = int((ctx["setpoint_delta"] < 0).sum())
    n_fan_down = int((ctx["fan_delta"]      < 0).sum())
    n_rec_on   = int((ctx["recirc_on"]   == True).sum())

    # Physics prior ordering by band
    acts = []
    if band == "hot":
        acts.append({"id":"sp_plus", "label":"Increase setpoint (+5 to +10 °C)", "est_W": sp_up,    "n": n_sp_up})
        acts.append({"id":"rec_on",  "label":"Turn Recirculation ON",            "est_W": recirc_on,"n": n_rec_on})
        acts.append({"id":"fan_down","label":"Reduce fan (−1 to −2)",            "est_W": fan_down, "n": n_fan_down})
    elif band == "cold":
        acts.append({"id":"sp_minus","label":"Lower setpoint (−5 to −10 °C)",    "est_W": sp_down,  "n": n_sp_down})
        acts.append({"id":"fan_down","label":"Reduce fan (−1 to −2)",            "est_W": fan_down, "n": n_fan_down})
        acts.append({"id":"rec_on",  "label":"Turn Recirculation ON",            "est_W": recirc_on,"n": n_rec_on})
    else:  # mild
        # pick the better of sp_up/sp_down from observed data
        best_is_up = (sp_up if (sp_up or -1e9) > (sp_down or -1e9) else sp_down) == sp_up
        best_sp    = sp_up if best_is_up else sp_down
        best_n     = n_sp_up if best_is_up else n_sp_down
        best_lbl   = "Increase setpoint (+)" if best_is_up else "Lower setpoint (−)"
        acts.append({"id":"sp_best","label":best_lbl, "est_W": best_sp, "n": best_n})
        acts.append({"id":"fan_down","label":"Reduce fan (−1 to −2)",    "est_W": fan_down, "n": n_fan_down})
        acts.append({"id":"rec_on",  "label":"Turn Recirculation ON",    "est_W": recirc_on,"n": n_rec_on})

    # If very little data, keep physics order 
    if n >= 5:
        def score(a):
            v = a["est_W"]
            return -1e9 if (v is None or _is_nan(v)) else float(v)
        acts = sorted(acts, key=score, reverse=True)

    return {
        "explanation": f"Nearest context: band={band}, speed={sb} • based on {n} past interactions ({confidence(n)})",
        "n": n,
        "actions": acts
    }
