from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pathlib, pickle

def try_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

@dataclass
class ModelSpec:
    key: str
    name: str
    needs_sequence: bool
    window: int
    feature_cols: List[str]

class BasePredictor:
    spec: ModelSpec
    def load(self): ...
    def predict_one_row(self, X_now: np.ndarray, seq: Optional[np.ndarray]=None) -> float: ...

class TorchSeqPredictor(BasePredictor):
    def __init__(self, key: str, name: str, model_path: pathlib.Path, feature_cols: List[str],
                 window: int, arch: str, scaler_path: Optional[pathlib.Path] = None,
                 target_scaler_path: Optional[pathlib.Path] = None):
        self.key = key
        self.name = name
        self.model_path = model_path
        self.feature_cols = feature_cols
        self.window = int(window)
        self.arch = arch
        self.scaler_path = scaler_path
        self.target_scaler_path = target_scaler_path
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.torch = try_torch()
        self.spec = ModelSpec(key=key, name=name, needs_sequence=True, window=self.window, feature_cols=feature_cols)

    def load(self):
        if self.torch is None:
            raise RuntimeError("PyTorch not installed.")
        from .seq_blocks import GRUHead, TinyTransformer
        input_dim = len(self.feature_cols)
        if self.arch == "gru":
            self.model = GRUHead(input_dim=input_dim)
        elif self.arch == "transformer":
            self.model = TinyTransformer(input_dim=input_dim)
        else:
            self.model = GRUHead(input_dim=input_dim)   # fallback
        state = self.torch.load(str(self.model_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        if self.scaler_path and self.scaler_path.exists():
            self.scaler = pickle.load(open(self.scaler_path, "rb"))
        if self.target_scaler_path and self.target_scaler_path.exists():
            self.target_scaler = pickle.load(open(self.target_scaler_path, "rb"))

    def _prep_seq(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[0] != self.window or seq.shape[1] != len(self.feature_cols):
            raise ValueError(f"Seq shape must be ({self.window}, {len(self.feature_cols)}). Got {seq.shape}.")
        if self.scaler is not None:
            seq = self.scaler.transform(seq)
        return seq.astype(np.float32)

    def predict_one_row(self, X_now: np.ndarray, seq: Optional[np.ndarray]=None) -> float:
        if seq is None:
            raise ValueError(f"{self.name} requires a sequence window.")
        x = self._prep_seq(seq)
        t = self.torch
        with t.no_grad():
            x_t = t.from_numpy(x).unsqueeze(0)  # (1,T,F)
            y_t = self.model(x_t)               # (1,T)
            last = y_t[:, -1]                   # (1,)
            y = float(last.cpu().numpy().squeeze())
            if self.target_scaler is not None:
                y = float(self.target_scaler.inverse_transform(np.array([[y]])).squeeze())
            return y

def build_registry(project_root: Optional[pathlib.Path] = None,
                   feature_cols: Optional[List[str]] = None,
                   window: int = 32) -> Dict[str, BasePredictor]:
    root = pathlib.Path(project_root or ".").resolve()
    m = root / "models"
    scaler = m / "input_scaler.pkl"
    tscaler = m / "target_scaler.pkl"
    cols = feature_cols or [
        # IMPORTANT: must match your DL training input order
        "speed", "ambient_temp", "cabin_temp", "fan", "recirc", "setpoint",
    ]
    return {
        "GRU (seq)": TorchSeqPredictor("gru", "GRU (seq)", m/"gru.pt", cols, window, "gru", scaler, tscaler),
        "Transformer (seq)": TorchSeqPredictor("transformer", "Transformer (seq)", m/"transformer.pt", cols, window, "transformer", scaler, tscaler),
        # Use GRU head for mamba.pt for now (it was trained as fallback)
        "Mamba (exp)": TorchSeqPredictor("mamba", "Mamba (exp)", m/"mamba.pt", cols, window, "gru", scaler, tscaler),
    }
