import pandas as pd

# Defaults (overridden by proxy_override from app)
TEMP_RPM_FEATURE = "voltage_drop"       # setpoint/cooling proxy
AIRFLOW_PROXY    = "rpm"                # fan proxy
HUM_OVER_VOLT    = "bat_voltage"        # recirc/electrical proxy

def _clamp(x, lo, hi): return max(lo, min(hi, x))

def _physics_multipliers(ambient_c: float, cabin_c: float,
                         setpoint_delta: int, fan_delta: int, recirc_on: bool,
                         sensitivity: float):
    """
    Returns (sp_scale, fan_scale, rec_scale) multipliers based on HVAC physics.
    - Hot band (≥28C): warmer setpoint saves; recirc helps more; fan adds small load
    - Cold band (≤12C): warmer setpoint costs (heating); recirc small help; fan adds small load
    - Mild: small effects
    """
    dT = cabin_c - ambient_c
    HOT, COLD = ambient_c >= 28, ambient_c <= 12

    # Base linear gains per "unit" of control; tuned conservatively
    # (These are *scales on proxies*, not watts; sensitivity amplifies visual change.)
    if HOT:
        # Cooling-dominant
        sp_scale  = 1.0 - 0.02 * setpoint_delta * sensitivity     # +ΔT (warmer) => scale < 1
        rec_scale = 1.0 - (0.03 if recirc_on else 0.0) * sensitivity
    elif COLD:
        # Heating-dominant: warmer setpoint should INCREASE energy
        sp_scale  = 1.0 + 0.02 * setpoint_delta * sensitivity     # +ΔT (warmer) => scale > 1
        rec_scale = 1.0 - (0.01 if recirc_on else 0.0) * sensitivity  # small benefit only
    else:
        # Mild: small effect
        sp_scale  = 1.0 - 0.01 * setpoint_delta * sensitivity
        rec_scale = 1.0 - (0.015 if recirc_on else 0.0) * sensitivity

    # Fan adds a bit of load in all bands (blower + exchange)
    fan_scale = 1.0 + 0.015 * fan_delta * sensitivity

    # Clamp to avoid wild swings in demos
    sp_scale  = _clamp(sp_scale, 0.80, 1.25)
    fan_scale = _clamp(fan_scale, 0.85, 1.20)
    rec_scale = _clamp(rec_scale, 0.90, 1.05 if COLD else 1.10)  # smaller effect in cold

    return sp_scale, fan_scale, rec_scale

def make_features(row_df: pd.DataFrame,
                  setpoint_delta: int,
                  fan_delta: int,
                  recirc_on: bool,
                  sensitivity: float,
                  predictor,
                  vehicle_type: str = "default",
                  proxy_override: dict | None = None,
                  ambient_col: str = "ambient_temp",
                  cabin_col: str = "cabin_temp") -> pd.DataFrame:
    """
    Physics-informed proxy scaling. Uses ambient/cabin to choose direction of effects.
    """
    df = row_df.copy()
    sp_col  = proxy_override.get("setpoint") if proxy_override else TEMP_RPM_FEATURE
    fan_col = proxy_override.get("fan")      if proxy_override else AIRFLOW_PROXY
    rh_col  = proxy_override.get("recirc")   if proxy_override else HUM_OVER_VOLT

    ambient = float(df.get(ambient_col, pd.Series([24])).iloc[0])
    cabin   = float(df.get(cabin_col,   pd.Series([24])).iloc[0])

    sp_scale, fan_scale, rec_scale = _physics_multipliers(
        ambient, cabin, setpoint_delta, fan_delta, recirc_on, sensitivity
    )

    # Apply scales (numeric only)
    if sp_col in df.columns:
        df[sp_col]  = pd.to_numeric(df[sp_col],  errors="coerce").fillna(0) * sp_scale
    if fan_col in df.columns:
        df[fan_col] = pd.to_numeric(df[fan_col], errors="coerce").fillna(0) * fan_scale
    if rh_col in df.columns:
        df[rh_col]  = pd.to_numeric(df[rh_col],  errors="coerce").fillna(0) * rec_scale

    # Expected-save (UI only): positive means "should save" per our physics rules
    # Use predictor effects just for guidance; sign depends on band.
    k_sp = predictor.effects.get("setpoint_slope_W_per_deg", 1.0)
    k_f  = predictor.effects.get("fan_slope_W_per_level", 0.4)
    k_r  = predictor.effects.get("recirc_gain_W", 2.0)
    veh  = predictor.scale_for_vehicle(vehicle_type)

    # Sign logic for UI hint
    hot, cold = ambient >= 28, ambient <= 12
    sgn_sp = -1 if hot else (1 if cold else -1)    # hotter: +Δ saves; colder: +Δ costs
    sgn_r  = -1 if hot else (-0.3 if cold else -0.5)  # smaller benefit in cold/mild
    expected_save_W = (k_sp * sgn_sp * abs(setpoint_delta)) \
                    + (k_f  * max(0, -fan_delta)) \
                    + (k_r  * (1 if recirc_on else 0) * (1.0 if hot else 0.4))
    df["_expected_save_W"] = float(expected_save_W) * float(veh)
    df["_proxy_cols"] = f"{sp_col}|{fan_col}|{rh_col}"
    return df
