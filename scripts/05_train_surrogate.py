"""Stage 4b: train the Hetero-GNN surrogate on the real counterfactual dataset (RTX 5070).

    .\\.venv\\Scripts\\python.exe scripts/05_train_surrogate.py                       # edgemlp + gatv2 ablation
    .\\.venv\\Scripts\\python.exe scripts/05_train_surrogate.py --conv edgemlp --epochs 40

Split (fixed before training, written into the metrics file):
  train      all NL patches except Rotterdam, plantings 0 .. N-11
  val        same patches, last 10 plantings (model selection only)
  test_city  Rotterdam (rot_*), all plantings       -> unseen city
  transfer   Freiburg (Overture heights), all plantings -> unseen city + lower-quality heights
Metrics: absolute UTCI MAE, and skill on the tree-induced CHANGE (vs the engine's unplanted
baseline, planting 0): correlation, MAE, and the zero-change reference MAE.
Outputs: data/models/heatgraph_<conv>.pt and data/models/metrics_<conv>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR, load_toml  # noqa: E402
from heatgraph.realscene import load_scene  # noqa: E402
from heatgraph.surrogate.graphs import LazyGraphDataset, SceneGraph  # noqa: E402
from heatgraph.surrogate.model import HeatGraphGNN  # noqa: E402
from heatgraph.surrogate.train import fit, predict  # noqa: E402

N_VAL = 10
# Leave-patch-out validation (v2 protocol): whole patches from the TRAINING cities, one per
# morphology family (open midrise / towers-industrial / suburban), never Rotterdam or Freiburg.
VAL_PATCHES = ["ams_bos_en_lommer", "ein_strijp_s", "amf_vathorst"]


def load_all(device, physics_edges=True):
    cfg = load_toml("engine.toml")
    out = {}
    for d in sorted((DATA_DIR / "processed").iterdir()):
        if not (d / "labels.npz").exists():
            continue
        s = load_scene(d)
        z = np.load(d / "labels.npz")
        X, U = z["plantings"], z["utci"]
        sg = SceneGraph(s, baseline_utci=U[0], cfg=cfg, device=device, physics_edges=physics_edges)
        out[d.name] = {"sg": sg, "X": X, "U": U}
    return out


def items(entry, idx):
    sg, X, U = entry["sg"], entry["X"], entry["U"]
    return [(sg, X[i], h, U[i, h]) for i in idx for h in range(sg.hours)]


def evaluate_split(model, entries, device):
    rows, per_patch = [], {}
    for pid, (e, idx) in entries.items():
        data = LazyGraphDataset(items(e, idx))
        pred = predict(model, list(data), device, batch_size=64)
        n, H, E = len(idx), e["sg"].hours, e["sg"].scene.n_segments
        P = pred.reshape(n, H, E)
        T = e["U"][idx]
        base = e["U"][0][None]
        d_true, d_pred = (T - base)[1:] if idx[0] == 0 else (T - base), (P - base)[1:] if idx[0] == 0 else (P - base)
        r = {"mae": float(np.mean(np.abs(P - T))), "bias": float(np.mean(P - T)),
             "change_mae": float(np.mean(np.abs(d_pred - d_true))), "zero_change_mae": float(np.mean(np.abs(d_true))),
             "change_corr": float(np.corrcoef(d_true.ravel(), d_pred.ravel())[0, 1]) if d_true.std() > 0 else None}
        per_patch[pid] = r
        rows.append((P - T, d_pred - d_true, d_true, d_pred))
    allP = np.concatenate([r[0].ravel() for r in rows])
    allD = np.concatenate([r[1].ravel() for r in rows])
    dt = np.concatenate([r[2].ravel() for r in rows])
    dp = np.concatenate([r[3].ravel() for r in rows])
    pooled = {"mae": float(np.mean(np.abs(allP))), "bias": float(np.mean(allP)), "change_mae": float(np.mean(np.abs(allD))),
              "zero_change_mae": float(np.mean(np.abs(dt))), "change_corr": float(np.corrcoef(dt, dp)[0, 1])}
    return pooled, per_patch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", action="append", choices=["edgemlp", "gatv2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-mode", choices=["patch", "planting"], default="patch",
                    help="patch: whole held-out training-city patches (v2); planting: last plantings (v1)")
    ap.add_argument("--features", choices=["v1", "v2"], default="v2", help="v2 adds crown-shadow/overhead edges")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all(device, physics_edges=(a.features == "v2"))
    test_city = [p for p in data if p.startswith("rot_")]
    transfer = [p for p in data if p.startswith("xfer_")]
    pool = [p for p in data if p not in test_city + transfer]
    n = {p: len(data[p]["X"]) for p in data}
    if a.val_mode == "patch":
        val_p = [p for p in VAL_PATCHES if p in pool]
        train = [p for p in pool if p not in val_p]
        tr_items = [it for p in train for it in items(data[p], range(n[p]))]
        va_items = [it for p in val_p for it in items(data[p], range(n[p]))]
        val_spec = {p: (data[p], list(range(n[p]))) for p in val_p}
    else:
        val_p, train = [], pool
        tr_items = [it for p in train for it in items(data[p], range(0, n[p] - N_VAL))]
        va_items = [it for p in train for it in items(data[p], range(n[p] - N_VAL, n[p]))]
        val_spec = {p: (data[p], list(range(n[p] - N_VAL, n[p]))) for p in train}
    split = {"train": train, "val_patches": val_p, "val_mode": a.val_mode, "test_city": test_city,
             "transfer": transfer, "n_train_graphs": len(tr_items), "n_val_graphs": len(va_items)}
    print(json.dumps(split), flush=True)
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for conv in a.conv or ["edgemlp", "gatv2"]:
        torch.manual_seed(0)
        name = f"{conv}_{a.features}_{a.val_mode}val{('_' + a.tag) if a.tag else ''}"
        model = HeatGraphGNN(hidden=a.hidden, layers=a.layers, slot_conv=conv, dropout=a.dropout)
        t = time.time()
        hist = fit(model, LazyGraphDataset(tr_items), list(LazyGraphDataset(va_items)), epochs=a.epochs, lr=a.lr,
                   batch_size=a.batch, device=device, seed=0, weight_decay=a.weight_decay, patience=a.patience)
        train_s = time.time() - t
        res = {"conv": conv, "name": name, "split": split, "train_seconds": round(train_s, 1), "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "hparams": vars(a), "history": hist}
        res["val"], res["val_per_patch"] = evaluate_split(model, val_spec, device)
        res["test_city"], res["test_city_per_patch"] = evaluate_split(
            model, {p: (data[p], list(range(n[p]))) for p in test_city}, device)
        if transfer:
            res["transfer"], res["transfer_per_patch"] = evaluate_split(
                model, {p: (data[p], list(range(n[p]))) for p in transfer}, device)
        torch.save({"state_dict": model.state_dict(), "conv": conv, "hidden": a.hidden, "layers": a.layers,
                    "dropout": a.dropout, "features": a.features, "split": split}, models_dir / f"heatgraph_{name}.pt")
        (models_dir / f"metrics_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        for k in ("val", "test_city", "transfer"):
            if k in res:
                r = res[k]
                print(f"[{name}] {k:9s} MAE {r['mae']:.3f} C bias {r['bias']:+.3f} | change corr {r['change_corr']:.3f} "
                      f"change MAE {r['change_mae']:.3f} vs zero-change {r['zero_change_mae']:.3f}", flush=True)
        print(f"[{name}] trained {train_s:.0f} s on {res['gpu']} | best epoch {hist.get('best_epoch')} "
              f"stopped {hist.get('stopped_epoch', len(hist['train_mae']))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
