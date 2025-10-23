from pathlib import Path
import json
import pandas as pd
from catboost import CatBoostRegressor


class Predictor:
    def __init__(self, version: str = "v1"):
        self.version = version
        m = Path("models")

        if version == "v2":
            # Point model (v2)
            self.model = CatBoostRegressor()
            self.model.load_model(str(m / "catboost_v2.cbm"))

            # Quantile models (P10 & P90)
            self.q10 = CatBoostRegressor()
            self.q10.load_model(str(m / "cb_q10.cbm"))

            self.q90 = CatBoostRegressor()
            self.q90.load_model(str(m / "cb_q90.cbm"))

            # Feature order for v2 (use v2 list if present, else fallback to v1)
            feat_file = m / "feature_columns_v2.json"
            if feat_file.exists():
                self.feature_cols = json.loads(feat_file.read_text())
            else:
                self.feature_cols = json.loads((m / "feature_columns.json").read_text())
            self.effects = self._load_effects()

        else:
            # v1 point model
            self.model = CatBoostRegressor()
            self.model.load_model(str(m / "catboost_model.cbm"))
            self.q10 = None
            self.q90 = None
            feat_file = m / "feature_columns.json"
            if feat_file.exists():
                self.feature_cols = json.loads(feat_file.read_text())
            else:
                # fallback to model feature names
                try:
                    self.feature_cols = list(self.model.feature_names_)
                except Exception:
                    self.feature_cols = []
            self.effects = self._load_effects()


    # (kept for compatibility—unused)
    def _load_q(self, path):
        p = Path(path)
        if p.exists():
            m = CatBoostRegressor()
            m.load_model(str(p))
            return m
        return None

    # Effects config (unused by predictor; preserved for compatibility)
    def _load_effects(self):
        defaults = {
            "setpoint_slope_W_per_deg": 0.9,
            "fan_slope_W_per_level": 0.4,
            "recirc_gain_W": 2.5,
            "per_vehicle_scale": {"default": 1.0, "ICE": 1.0, "HEV": 1.05, "PHEV": 1.10, "EV": 1.20},
        }
        p = Path("ac_advisor/effects.json")
        if p.exists():
            try:
                defaults.update(json.loads(p.read_text()))
            except Exception:
                pass
        return defaults

    def scale_for_vehicle(self, vehicle_type: str) -> float:
        # ICE-only dataset → no per-vehicle scaling
        return 1.0

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.feature_cols)
        return X.reindex(columns=cols, fill_value=0)

    def predict(self, X: pd.DataFrame):
        X2 = self._align(X)
        y_hat = pd.Series(self.model.predict(X2), index=X2.index)

        lo = hi = None
        if self.q10 is not None and self.q90 is not None:
            lo = pd.Series(self.q10.predict(X2), index=X2.index)
            hi = pd.Series(self.q90.predict(X2), index=X2.index)

        return y_hat, (lo, hi)

    def predict_quantiles(self, X: pd.DataFrame, qs=(0.10, 0.90)):
        X2 = self._align(X)
        if self.q10 is None or self.q90 is None:
            raise RuntimeError("Quantile models not loaded for this predictor.")
        out = {}
        for q in qs:
            if abs(q - 0.10) < 1e-6:
                out[q] = list(self.q10.predict(X2))
            elif abs(q - 0.90) < 1e-6:
                out[q] = list(self.q90.predict(X2))
            else:
                # Only P10/P90 available
                raise ValueError(f"Unsupported quantile {q}. Only 0.10 and 0.90 are available.")
        return out
