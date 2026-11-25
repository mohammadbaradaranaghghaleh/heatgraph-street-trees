"""Correlation between the summed v2 crown-shadow edge feature and the engine's tree-induced
change in segment UTCI (no model involved). Saved for the manuscript's audit trail.

    .\\.venv\\Scripts\\python.exe scripts/11_feature_correlation.py  -> data/models/physics_feature_corr.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR, load_toml  # noqa: E402
from heatgraph.realscene import load_scene  # noqa: E402
from heatgraph.surrogate.graphs import EDGE_FEATURES, SceneGraph  # noqa: E402

SHADE, OVER = EDGE_FEATURES.index("crown_shade_frac"), EDGE_FEATURES.index("overhead_frac")


def main():
    cfg = load_toml("engine.toml")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for pid in ("rot_centrum", "ams_bijlmer"):
        s = load_scene(DATA_DIR / "processed" / pid)
        z = np.load(DATA_DIR / "processed" / pid / "labels.npz")
        X, U = z["plantings"], z["utci"]
        sg = SceneGraph(s, baseline_utci=U[0], cfg=cfg, device=dev)
        E = s.n_segments
        xs, xo, ys = [], [], []
        for i in range(1, len(X)):
            keep = X[i][sg.link_slot] > 0
            for h in range(sg.hours):
                xs.append(np.bincount(sg.link_seg[keep], sg.link_attr[h][keep, SHADE], E))
                xo.append(np.bincount(sg.link_seg[keep], sg.link_attr[h][keep, OVER], E))
                ys.append(U[i, h] - U[0, h])
        xs, xo, ys = map(np.concatenate, (xs, xo, ys))
        out[pid] = {"corr_sum_crown_shade_vs_dUTCI": float(np.corrcoef(xs, ys)[0, 1]),
                    "corr_sum_overhead_vs_dUTCI": float(np.corrcoef(xo, ys)[0, 1]), "n": int(len(ys))}
        print(pid, out[pid])
    (DATA_DIR / "models" / "physics_feature_corr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
