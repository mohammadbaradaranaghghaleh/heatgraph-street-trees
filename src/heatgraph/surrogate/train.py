"""Training / evaluation loops (CPU in this phase; pass device='cuda' later on purpose only)."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .model import HeatGraphGNN


def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(model: HeatGraphGNN, train_data: list, val_data: list | None = None, epochs: int = 50, lr: float = 3e-3,
        batch_size: int = 16, weight_decay: float = 1e-5, device: str = "cpu", seed: int = 0, verbose: bool = False,
        patience: int | None = None):
    """Train with MSE on standardised targets; returns history dict. Restores the best-val weights.
    patience: stop after this many epochs without validation improvement (None = run all epochs)."""
    seed_everything(seed)
    dev = torch.device(device)
    ys = torch.cat([d["segment"].y - (d["segment"].base if "base" in d["segment"] else 0.0) for d in train_data])
    model.y_mean.fill_(float(ys.mean())); model.y_std.fill_(float(ys.std().clamp_min(1e-3)))
    model.to(dev)
    with torch.no_grad():                                 # materialise lazy shapes before the optimiser
        model.predict(train_data[0].clone().to(dev))      # .to() is IN-PLACE for HeteroData: copy first
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    hist, best, best_state = {"train_mae": [], "val_mae": []}, float("inf"), None
    for ep in range(epochs):
        model.train()
        err, n = 0.0, 0
        for batch in loader:
            batch = batch.to(dev)
            opt.zero_grad()
            pred = model.predict(batch)
            loss = torch.mean(((pred - batch["segment"].y) / model.y_std) ** 2)
            loss.backward()
            opt.step()
            err += float(torch.abs(pred - batch["segment"].y).sum()); n += batch["segment"].y.numel()
        sched.step()
        hist["train_mae"].append(err / n)
        if val_data:
            v = evaluate(model, val_data, device)["mae"]
            hist["val_mae"].append(v)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
                hist["best_epoch"] = ep + 1
            elif patience is not None and ep + 1 - hist.get("best_epoch", 0) >= patience:
                hist["stopped_epoch"] = ep + 1
                break
        if verbose:
            print(ep, hist["train_mae"][-1], hist["val_mae"][-1] if val_data else "")
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict(model, data_list: list, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """Concatenated segment predictions in data_list order."""
    model.eval()
    out = []
    for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
        out.append(model.predict(batch.to(torch.device(device))).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, data_list: list, device: str = "cpu") -> dict:
    y = torch.cat([d["segment"].y for d in data_list]).numpy()
    p = predict(model, data_list, device)
    return {"mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "bias": float(np.mean(p - y)), "pred": p, "true": y}
