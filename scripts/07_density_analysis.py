"""Where does the GNN beat the linear physics baseline? Change error by planting density.

    .\\.venv\\Scripts\\python.exe scripts/07_density_analysis.py

Uses the trained v2 GNN (data/models/heatgraph_edgemlp_v2_patchval.pt) and the per-hour
linear baseline (refitted exactly as in 06_baselines.py) on Rotterdam + Freiburg, grouped by
the share of slots planted. Writes data/models/density_analysis.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR  # noqa: E402
from heatgraph.surrogate.graphs import LazyGraphDataset  # noqa: E402
from heatgraph.surrogate.model import HeatGraphGNN  # noqa: E402
from heatgraph.surrogate.train import predict  # noqa: E402

train_mod = __import__("05_train_surrogate")
base_mod = __import__("06_baselines")
BINS = [0.0, 0.10, 0.20, 0.30, 0.41]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = train_mod.load_all(dev, physics_edges=True)
    ck = torch.load(DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt", map_location=dev)
    model = HeatGraphGNN(hidden=ck["hidden"], layers=ck["layers"], dropout=ck["dropout"]).to(dev)
    first = next(iter(data.values()))
    with torch.no_grad():
        model.predict(first["sg"].build(first["X"][0], 0).to(dev))      # materialise lazy shapes
    model.load_state_dict(ck["state_dict"])
    model.eval()

    split = ck["split"]
    D = {p: base_mod.design(data[p], range(1, len(data[p]["X"]))) for p in data}
    H = data[split["train"][0]]["sg"].hours
    coef = np.array([np.linalg.lstsq(np.concatenate([D[p][0][:, h].reshape(-1, 3) for p in split["train"]]),
                                     np.concatenate([D[p][1][:, h].ravel() for p in split["train"]]), rcond=None)[0]
                     for h in range(H)])
    rows = []
    for p in split["test_city"] + split["transfer"]:
        e = data[p]
        idx = list(range(1, len(e["X"])))
        P = predict(model, list(LazyGraphDataset(train_mod.items(e, idx))), dev, 64).reshape(len(idx), H, -1)
        d_gnn = P - e["U"][0][None]
        d_lin = np.einsum("nhef,hf->nhe", D[p][0], coef)
        d_true = D[p][1]
        dens = e["X"][idx].mean(1)
        for j in range(len(idx)):
            rows.append((p, dens[j], np.abs(d_gnn[j] - d_true[j]).mean(), np.abs(d_lin[j] - d_true[j]).mean(),
                         np.abs(d_true[j]).mean()))
    out = {"bins": BINS, "by_bin": []}
    print("density bin   n   GNN change MAE   linear change MAE   zero-change   GNN gain")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = [r for r in rows if lo <= r[1] < hi]
        if not sel:
            continue
        g, l, z = (float(np.mean([r[k] for r in sel])) for k in (2, 3, 4))
        out["by_bin"].append({"lo": lo, "hi": hi, "n": len(sel), "gnn": g, "linear": l, "zero": z})
        print(f"[{lo:.2f},{hi:.2f})  {len(sel):3d}   {g:.3f}            {l:.3f}               {z:.3f}        {100 * (l - g) / l:+.0f} %")
    (DATA_DIR / "models" / "density_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
