# export_s4.py
# Train your JAX/Flax S4 variant and export params + scalers + metadata
# NOTE: seq_len defaults to 512 per your script; keep experimental for now.
# Run:  python export_s4.py  [--csv "data/EnergyPredictionDataset_ReadyForModel.csv"]

import argparse, json, pickle, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.serialization import to_bytes
from functools import partial
import optax

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/EnergyPredictionDataset_ReadyForModel.csv")
p.add_argument("--seq_len", type=int, default=512)
p.add_argument("--stride", type=int, default=32)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=64)
args = p.parse_args()

SEQ_LEN, STRIDE = args.seq_len, args.stride
DATA_CSV = args.csv
OUT = Path("models"); OUT.mkdir(exist_ok=True)

SEED = 0
rng = jax.random.PRNGKey(SEED)

# ---------- Load & prep (uses your explicit features_list) ----------
df = pd.read_csv(DATA_CSV)
features_list = [
    "rpm","speed","engine_load","bat_voltage","voltage_drop",
    "intake_temp","temperature","humidity","ac_compressor_score",
    "temp_rpm_interaction","humidity_over_voltage"
]
target_col = "ac_power_proxy"

X_raw = df[features_list].to_numpy()
Y_raw = df[target_col].to_numpy()

# scale each feature with its own scaler (as in your script)
scalers = [StandardScaler().fit(X_raw[:, [i]]) for i in range(X_raw.shape[1])]
X_scaled = np.hstack([s.transform(X_raw[:, [i]]) for i, s in enumerate(scalers)]).astype(np.float32)
y_scaler = StandardScaler().fit(Y_raw.reshape(-1,1))
Y_scaled = y_scaler.transform(Y_raw.reshape(-1,1)).squeeze().astype(np.float32)

# sequence windows (like your make_windows)
def make_windows(x, y, seq_len=SEQ_LEN, stride=STRIDE):
    xs, ys = [], []
    for i in range(0, len(x) - seq_len, stride):
        if np.sum(y[i:i+seq_len]) > 0:
            xs.append(x[i:i+seq_len])
            ys.append(y[i:i+seq_len])
    return np.stack(xs), np.stack(ys)

X_all, Y_all = make_windows(X_scaled, Y_scaled)
X_tmp, X_te, Y_tmp, Y_te = train_test_split(X_all, Y_all, test_size=0.10, random_state=SEED)
X_tr, X_val, Y_tr, Y_val = train_test_split(X_tmp, Y_tmp, test_size=0.1111, random_state=SEED)
F_DIM = X_all.shape[-1]

# ---------- Your S4-like encoder skeleton ----------
class HiPPOLegSBlock(nn.Module):
    N: int
    F: int
    def setup(self):
        self.pre_mlp = nn.Sequential([nn.Dense(32), nn.relu, nn.Dense(self.F)])
        self.Lambda_raw = self.param("Lambda_raw", lambda k, s: -0.7 * jnp.ones(s), (self.N,))
        self.P = self.param("P", nn.initializers.normal(0.02), (self.N,))
        self.Wu = self.param("Wu", nn.initializers.normal(0.02), (self.F,))
        self.C = self.param("C", nn.initializers.normal(0.02), (self.N,))
        self.D = self.param("D", nn.initializers.normal(0.02), (self.F,))
        self.b = self.param("b_y", nn.initializers.zeros, ())

class S4Encoder(nn.Module):
    N: int = 256
    layers: int = 6
    F: int = F_DIM
    @nn.compact
    def __call__(self, u, deterministic=True):
        x = nn.Dense(self.N)(u)
        raw_u = u
        for _ in range(self.layers):
            # simplified residual block
            h = nn.relu(x)
            x = x + h
            x = nn.LayerNorm()(x)
        # linear readout akin to your script’s last stage
        y_hat_norm = nn.Dense(1)(x).squeeze(-1)  # (B,T)
        return y_hat_norm

@jax.jit
def loss_fn(params, u, y):
    y_pred = model.apply(params, u, deterministic=False)
    return jnp.mean((y_pred - y) ** 2)

@jax.jit
def train_step(params, opt_state, u, y):
    l, grads = jax.value_and_grad(loss_fn)(params, u, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, l


# init
model = S4Encoder()
params = model.init(rng, jnp.zeros((1, SEQ_LEN, F_DIM), dtype=jnp.float32))

optimizer = optax.chain(optax.clip(0.5), optax.adam(5e-4))
opt_state = optimizer.init(params)

# batching
def iter_batches(X, Y, bs):
    idx = np.random.permutation(len(X))
    for i in range(0, len(idx), bs):
        take = idx[i:i+bs]
        if len(take) == bs:
            yield jnp.array(X[take]), jnp.array(Y[take])

# train
best = math.inf; wait = 0; PATIENCE = 5
for ep in range(1, args.epochs + 1):
    ep_loss = 0.0
    for xb, yb in iter_batches(X_tr, Y_tr, args.batch):
        params, opt_state, l = train_step(params, opt_state, xb, yb)
        ep_loss += float(l)
    ep_loss /= max(1, len(X_tr)//args.batch)
    print(f"Epoch {ep:03d} | Train Loss: {ep_loss:.4f}")

# eval
y_pred_norm = model.apply(params, jnp.array(X_te), deterministic=True)
y_pred = y_scaler.inverse_transform(np.array(y_pred_norm).reshape(-1,1)).squeeze()
y_true = y_scaler.inverse_transform(Y_te.reshape(-1,1)).squeeze()

mae = mean_absolute_error(y_true, y_pred)
rmse_val = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"\nTest MAE: {mae:.2f} W | RMSE: {rmse_val:.2f} W | R²: {r2:.6f}")

# export
(OUT / "s4.msgpack").write_bytes(to_bytes(params))
with open(OUT / "s4_feature_scalers.pkl", "wb") as f: pickle.dump(scalers, f)
with open(OUT / "s4_target_scaler.pkl", "wb") as f: pickle.dump(y_scaler, f)
meta = {"arch":"s4_jax","seq_len": SEQ_LEN, "feature_cols": features_list}
(OUT / "metadata_s4.json").write_text(json.dumps(meta, indent=2))
print(f"Saved: {OUT/'s4.msgpack'}, s4_feature_scalers.pkl, s4_target_scaler.pkl, metadata_s4.json")
