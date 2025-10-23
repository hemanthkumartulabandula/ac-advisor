# export_gru.py
# Train GRU (seq_len=32) and export weights + scalers + metadata
# Run:  python export_gru.py  [--csv "data/EnergyPredictionDataset_ReadyForModel.csv"]

import argparse, json, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/EnergyPredictionDataset_ReadyForModel.csv")
p.add_argument("--seq_len", type=int, default=32)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

SEQ_LEN = args.seq_len
DATA_CSV = args.csv
OUT = Path("models"); OUT.mkdir(exist_ok=True)

# ---------- Load & prep ----------
df = pd.read_csv(DATA_CSV)
df.drop(columns=["timestamp", "timestamp_rounded", "weather_desc"], errors="ignore", inplace=True)

target_col = "ac_power_proxy"
feature_cols = sorted([c for c in df.columns if c != target_col])

X = df[feature_cols].values.astype(np.float32)
y = df[[target_col]].values.astype(np.float32)

# scale features + target
input_scaler = StandardScaler().fit(X)
X_scaled = input_scaler.transform(X).astype(np.float32)

target_scaler = StandardScaler().fit(y)
y_scaled = target_scaler.transform(y).astype(np.float32).squeeze(-1)

# make sequences
n_seq = len(X_scaled) // SEQ_LEN
X_seq = X_scaled[: n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN, -1)
y_seq = y_scaled[: n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN)

X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32)

class SeqDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SeqDS(X_tr_t, y_tr_t), batch_size=args.batch, shuffle=True)

# ---------- Model ----------
class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):           # x: (B,T,F)
        out, _ = self.gru(x)
        return self.fc(out).squeeze(-1)  # (B,T)

model = GRUHead(input_dim=X_tr.shape[2])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---------- Train ----------
for ep in range(1, args.epochs + 1):
    model.train()
    tot = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item()
    print(f"Epoch {ep:03d} | Train loss: {tot / max(1,len(train_loader)):.4f}")

# ---------- Eval (inverse target scaling) ----------
model.eval()
with torch.no_grad():
    pred_scaled = model(X_te_t).numpy().flatten()
true_scaled = y_te_t.numpy().flatten()

pred = target_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
true = target_scaler.inverse_transform(true_scaled.reshape(-1,1)).flatten()

print("\n--- Test Metrics (GRU) ---")
print(f"MAE:  {mean_absolute_error(true, pred):.4f} W")
print(f"RMSE: {rmse(true, pred):.4f} W")
print(f"R²:   {r2_score(true, pred):.6f}")

# ---------- Export ----------
torch.save(model.state_dict(), OUT / "gru.pt")
with open(OUT / "input_scaler.pkl", "wb") as f: pickle.dump(input_scaler, f)
with open(OUT / "target_scaler.pkl", "wb") as f: pickle.dump(target_scaler, f)
meta = {
    "arch": "gru",
    "seq_len": SEQ_LEN,
    "feature_cols": feature_cols,
}
(OUT / "metadata_gru.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved: {OUT/'gru.pt'}, input_scaler.pkl, target_scaler.pkl, metadata_gru.json")
