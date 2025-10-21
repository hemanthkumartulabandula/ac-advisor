from pathlib import Path
import json, pandas as pd
from catboost import CatBoostRegressor

class Predictor:
    def __init__(self, version="v1"):
        self.version = version
        if version == "v2" and Path("models/catboost_v2.cbm").exists():
            self.model = CatBoostRegressor(); self.model.load_model("models/catboost_v2.cbm")
            self.feature_cols = json.loads(Path("models/feature_columns_v2.json").read_text())
        else:
            self.model = CatBoostRegressor(); self.model.load_model("models/catboost_model.cbm")
            self.feature_cols = json.loads(Path("models/feature_columns.json").read_text())
        # optional quantile models for intervals
        self.q10 = self._load_q("models/cb_q10.cbm")
        self.q90 = self._load_q("models/cb_q90.cbm")
        self.effects = self._load_effects()

    def _load_q(self, path):
        p = Path(path)
        if p.exists():
            m = CatBoostRegressor(); m.load_model(str(p)); return m
        return None

    def _load_effects(self):
        defaults = {
            "setpoint_slope_W_per_deg": 0.9,
            "fan_slope_W_per_level": 0.4,
            "recirc_gain_W": 2.5,
            "per_vehicle_scale": {"default":1.0,"ICE":1.0,"HEV":1.05,"PHEV":1.10,"EV":1.20}
        }
        p = Path("ac_advisor/effects.json")
        if p.exists():
            try: defaults.update(json.loads(p.read_text()))
            except Exception: pass
        return defaults

    def scale_for_vehicle(self, vehicle_type: str) -> float:
        # ICE-only dataset → no per-vehicle scaling
        return 1.0


    def predict(self, X_df: pd.DataFrame):
        X = X_df.reindex(columns=self.feature_cols, fill_value=0)
        y = self.model.predict(X)
        lo = self.q10.predict(X) if self.q10 else None
        hi = self.q90.predict(X) if self.q90 else None
        return pd.Series(y, index=X_df.index), (lo, hi)
