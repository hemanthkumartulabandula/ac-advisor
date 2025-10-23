# export_transformer.py
# Train Transformer (seq_len=32) and export weights + scalers + metadata
# Run:  python export_transformer.py  [--csv "data/EnergyPredictionDataset_ReadyForModel.csv"]

import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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
df.drop(columns=['timestamp','timestamp_rounded','weather_desc'], errors='ignore', inplace=True)

target_col = "ac_power_proxy"
feature_cols = sorted([c for c in df.columns if c != target_col])

X = df[feature_cols].values.astype(np.float32)
y = df[[target_col]].values.astype(np.float32)

input_scaler = StandardScaler().fit(X)
X_scaled = input_scaler.transform(X).astype(np.float32)

target_scaler = StandardScaler().fit(y)
y_scaled = target_scaler.transform(y).astype(np.float32).squeeze(-1)

n_seq = len(X_scaled) // SEQ_LEN
X_seq = X_scaled[:n_seq*SEQ_LEN].reshape(n_seq, SEQ_LEN, -1)
y_seq = y_scaled[:n_seq*SEQ_LEN].reshape(n_seq, SEQ_LEN)

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

# ---------- Positional encoding ----------
def sinusoid_encoding(seq_len, d_model, device):
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    i   = torch.arange(0, d_model, 2, device=device)
    rates = 1 / torch.pow(10000, i / d_model)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * rates)
    pe[:, 1::2] = torch.cos(pos * rates)
    return pe.unsqueeze(0)  # (1,T,D)

# ---------- Model ----------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer("pe", sinusoid_encoding(SEQ_LEN, d_model, device=torch.device("cpu")))

    def forward(self, x):
        z = self.in_proj(x) + self.pe[:, :x.size(1), :]
        z = self.enc(z)
        return self.fc(z).squeeze(-1)

model = TransformerModel(input_dim=X_tr.shape[2])
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

# ---------- Eval ----------
model.eval()
with torch.no_grad():
    pred_scaled = model(X_te_t).numpy().flatten()
true_scaled = y_te_t.numpy().flatten()

pred = target_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
true = target_scaler.inverse_transform(true_scaled.reshape(-1,1)).flatten()

mae = mean_absolute_error(true, pred)
rmse_val = root_mean_squared_error(true, pred)
r2 = r2_score(true, pred)
print("\n--- Test Metrics (Transformer) ---")
print(f"MAE: {mae:.4f} W  |  RMSE: {rmse_val:.4f} W  |  R²: {r2:.6f}")

# ---------- Export ----------
torch.save(model.state_dict(), OUT / "transformer.pt")
with open(OUT / "input_scaler.pkl", "wb") as f: pickle.dump(input_scaler, f)
with open(OUT / "target_scaler.pkl", "wb") as f: pickle.dump(target_scaler, f)
meta = {
    "arch": "transformer",
    "seq_len": SEQ_LEN,
    "feature_cols": feature_cols,
}
(OUT / "metadata_transformer.json").write_text(json.dumps(meta, indent=2))
print(f"Saved: {OUT/'transformer.pt'}, input_scaler.pkl, target_scaler.pkl, metadata_transformer.json")

