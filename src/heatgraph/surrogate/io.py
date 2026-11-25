"""Load trained surrogates."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import HeatGraphGNN


def load_gnn(path, example_data, device: str = "cpu") -> HeatGraphGNN:
    """Rebuild a HeatGraphGNN from a checkpoint written by scripts/05_train_surrogate.py.
    `example_data` (one HeteroData) materialises lazily-shaped layers before loading."""
    ck = torch.load(Path(path), map_location=device)
    m = HeatGraphGNN(hidden=ck["hidden"], layers=ck["layers"], dropout=ck.get("dropout", 0.0),
                     slot_conv=ck.get("conv", "edgemlp")).to(device)
    with torch.no_grad():
        m.predict(example_data.clone().to(device))
    m.load_state_dict(ck["state_dict"])
    return m.eval()
