# ac_advisor/s4_infer_jax.py
from __future__ import annotations

import json, pickle
from pathlib import Path
from typing import List, Optional
import numpy as np

# JAX/Flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.serialization import from_bytes

# ---------- Minimal S4-like encoder to match export_s4.py ----------
class S4Encoder(nn.Module):
    N: int         # latent width (e.g., 256)
    layers: int    # number of residual blocks (e.g., 6)
    F: int         # input feature dim

    @nn.compact
    def __call__(self, u, deterministic: bool = True):
        # u: (B, T, F)
        x = nn.Dense(self.N)(u)
        for _ in range(self.layers):
            h = nn.relu(x)
            x = x + h
            x = nn.LayerNorm()(x)
        y_hat_norm = nn.Dense(1)(x).squeeze(-1)  # (B, T)
        return y_hat_norm

class S4JaxPredictor:
    """
    Loads params saved by export_s4.py:
      - models/s4.msgpack            (Flax params via to_bytes)
      - models/s4_feature_scalers.pkl (list[StandardScaler] per feature)
      - models/s4_target_scaler.pkl   (StandardScaler for target)
      - models/metadata_s4.json       (contains seq_len and feature_cols)
    Provides: predict_one_row(X_now, seq) -> float (last-timestep prediction, inverse-scaled).
    """
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = Path(models_dir)
        self.params = None
        self.feature_cols: List[str] = []
        self.seq_len: int = 512
        self._feature_scalers = None  # list of sklearn scalers
        self._target_scaler = None
        self._model: Optional[S4Encoder] = None
        self._apply = None  # jitted apply fn

    @property
    def name(self) -> str:
        return "S4 (JAX)"

    def available(self) -> bool:
        return (self.models_dir/"s4.msgpack").exists() and \
               (self.models_dir/"metadata_s4.json").exists() and \
               (self.models_dir/"s4_feature_scalers.pkl").exists() and \
               (self.models_dir/"s4_target_scaler.pkl").exists()

    def load(self):
        meta = json.loads((self.models_dir/"metadata_s4.json").read_text())
        self.feature_cols = list(meta.get("feature_cols", []))
        self.seq_len = int(meta.get("seq_len", 512))
        # You can tune these to your export settings if you changed them
        N = int(meta.get("N", 256))          # default used in export script
        layers = int(meta.get("layers", 6))  # default used in export script
        F = len(self.feature_cols)

        # Build model and init params for shape, then load bytes
        self._model = S4Encoder(N=N, layers=layers, F=F)
        # Dummy init to get a params tree of the right shape
        dummy = jnp.zeros((1, self.seq_len, F), dtype=jnp.float32)
        params0 = self._model.init(jax.random.PRNGKey(0), dummy, deterministic=True)
        bytes_blob = (self.models_dir/"s4.msgpack").read_bytes()
        self.params = from_bytes(params0, bytes_blob)

        # Load scalers
        with open(self.models_dir/"s4_feature_scalers.pkl", "rb") as f:
            self._feature_scalers = pickle.load(f)
        with open(self.models_dir/"s4_target_scaler.pkl", "rb") as f:
            self._target_scaler = pickle.load(f)

        # JIT apply to keep it fast
        def _apply_fn(params, x):
            return self._model.apply(params, x, deterministic=True)  # (B, T)
        self._apply = jax.jit(_apply_fn)

    def _scale_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: (T, F) in the SAME feature order as self.feature_cols.
        Applies per-feature StandardScaler.
        """
        if seq.ndim != 2 or seq.shape[1] != len(self.feature_cols):
            raise ValueError(f"seq must be (T,{len(self.feature_cols)}); got {seq.shape}")
        if self._feature_scalers is None:
            return seq.astype(np.float32)
        # each scaler transforms its single column
        cols = []
        for i, sc in enumerate(self._feature_scalers):
            col = sc.transform(seq[:, [i]])  # (T,1)
            cols.append(col.astype(np.float32))
        scaled = np.hstack(cols).astype(np.float32)  # (T,F)
        return scaled

    def predict_one_row(self, X_now, seq: np.ndarray) -> float:
        """
        Takes the last-timestep prediction (inverse-scaled to W).
        X_now is unused (kept for interface symmetry).
        """
        if self.params is None:
            raise RuntimeError("Call load() first.")
        # Ensure length == seq_len (left-pad if shorter)
        T, F = seq.shape
        if T < self.seq_len:
            pad = np.repeat(seq[:1, :], self.seq_len - T, axis=0)
            seq_use = np.vstack([pad, seq])
        else:
            seq_use = seq[-self.seq_len:, :]

        seq_use = self._scale_sequence(seq_use)  # (T,F)
        x = jnp.array(seq_use[None, :, :], dtype=jnp.float32)  # (1,T,F)
        y_norm = self._apply(self.params, x)  # (1,T)
        last_norm = np.array(y_norm)[0, -1]
        if self._target_scaler is not None:
            last = float(self._target_scaler.inverse_transform(np.array([[last_norm]])).squeeze())
        else:
            last = float(last_norm)
        return last
