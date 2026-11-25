"""Non-GNN baseline on the v2 physics features: does the GNN add anything?

    .\\.venv\\Scripts\\python.exe scripts/06_baselines.py

Per-hour linear model of the tree-induced change on each segment:
    dUTCI[seg, h] ~ a_h * sum(crown_shade_frac) + b_h * sum(overhead_frac) + c_h * n_planted_links
fitted by least squares on the GNN's TRAINING patches only (same split as 05_train_surrogate.py,
leave-patch-out), evaluated on the validation patches, Rotterdam and Freiburg with the same
change metrics. Writes data/models/metrics_linear_physics.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR  # noqa: E402
from heatgraph.surrogate.graphs import EDGE_FEATURES  # noqa: E402

train_mod = __import__("05_train_surrogate")
SHADE, OVER = EDGE_FEATURES.index("crown_shade_frac"), EDGE_FEATURES.index("overhead_frac")


def design(entry, idx):
    sg, X, U = entry["sg"], entry["X"], entry["U"]
    E = sg.scene.n_segments
    rows, ys = [], []
    for i in idx:
        keep = X[i][sg.link_slot] > 0
        seg = sg.link_seg[keep]
        n_links = np.bincount(seg, minlength=E)
        per_h = []
        for h in range(sg.hours):
            a = sg.link_attr[h][keep]
            per_h.append(np.column_stack([np.bincount(seg, a[:, SHADE], E), np.bincount(seg, a[:, OVER], E), n_links]))
        rows.append(np.stack(per_h))                       # (hours, E, 3)
        ys.append(U[i] - U[0])                              # (hours, E)
    return np.stack(rows), np.stack(ys)                     # (n, hours, E, 3), (n, hours, E)


def metrics(dp, dt):
    return {"change_mae": float(np.mean(np.abs(dp - dt))), "zero_change_mae": float(np.mean(np.abs(dt))),
            "change_corr": float(np.corrcoef(dt.ravel(), dp.ravel())[0, 1])}


def main():
    data = train_mod.load_all("cuda" if torch.cuda.is_available() else "cpu", physics_edges=True)
    test_city = [p for p in data if p.startswith("rot_")]
    transfer = [p for p in data if p.startswith("xfer_")]
    pool = [p for p in data if p not in test_city + transfer]
    val_p = [p for p in train_mod.VAL_PATCHES if p in pool]
    train = [p for p in pool if p not in val_p]
    D = {p: design(data[p], range(1, len(data[p]["X"]))) for p in data}
    H = data[train[0]]["sg"].hours
    coef = []
    for h in range(H):
        A = np.concatenate([D[p][0][:, h].reshape(-1, 3) for p in train])
        y = np.concatenate([D[p][1][:, h].ravel() for p in train])
        coef.append(np.linalg.lstsq(A, y, rcond=None)[0])
    coef = np.array(coef)                                   # (hours, 3)

    def predict(p):
        F, _ = D[p]
        return np.einsum("nhef,hf->nhe", F, coef)

    out = {"coef_per_hour": coef.tolist(), "split": {"train": train, "val": val_p, "test_city": test_city,
                                                       "transfer": transfer}}
    for name, ps in (("val", val_p), ("test_city", test_city), ("transfer", transfer)):
        dp = np.concatenate([predict(p).ravel() for p in ps])
        dt = np.concatenate([D[p][1].ravel() for p in ps])
        out[name] = metrics(dp, dt)
        r = out[name]
        print(f"[linear-physics] {name:9s} change MAE {r['change_mae']:.3f} vs zero {r['zero_change_mae']:.3f} "
              f"| change corr {r['change_corr']:.3f}")
    (DATA_DIR / "models" / "metrics_linear_physics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
