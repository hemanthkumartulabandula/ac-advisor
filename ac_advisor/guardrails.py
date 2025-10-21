# ac_advisor/guardrails.py
import pandas as pd

def defog_risk(row: pd.Series) -> bool:
    # Conservative: risk if any of these indicate fog likelihood
    fog_flag = bool(row.get("windshield_fog_flag", False))
    ch = row.get("cabin_humidity", None)
    dew = row.get("dew_point", None)
    cab = row.get("cabin_temp", None)
    near_dew = False
    try:
        near_dew = (dew is not None and cab is not None and abs(float(cab) - float(dew)) <= 1.5)
    except Exception:
        near_dew = False
    high_rh = False
    try:
        high_rh = (ch is not None and float(ch) >= 85.0)
    except Exception:
        high_rh = False
    return fog_flag or near_dew or high_rh

def comfort_guard(ambient: float, cabin: float, setpoint_delta: int) -> bool:
    # Block extreme ΔT recommendations
    dT = cabin - ambient
    projected = dT - setpoint_delta
    return abs(projected) > 15.0
