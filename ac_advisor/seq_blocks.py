from __future__ import annotations
import torch
import torch.nn as nn

class GRUHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):            # x: (B,T,F)
        out, _ = self.gru(x)
        return self.fc(out).squeeze(-1)  # (B,T)

class TinyTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc  = nn.Linear(d_model, 1)
    def forward(self, x):            # x: (B,T,F)
        z = self.in_proj(x)
        z = self.enc(z)
        return self.fc(z).squeeze(-1)    # (B,T)
