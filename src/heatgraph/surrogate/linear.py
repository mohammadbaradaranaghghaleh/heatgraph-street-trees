"""Linear physics surrogate: a cheap, transparent baseline with the GNN's interface.

    dUTCI[seg, h] = a_h * sum(crown_shade_frac) + b_h * sum(overhead_frac) + c_h * n_planted_links
    UTCI[seg, h]  = baseline_engine[seg, h] + dUTCI[seg, h]

Sums run over the planted slot->segment links of the segment; coefficients are per hour index
(all patches share the same forcing schedule: 08-19 local on the three heatwave days).
`predict(batch)` takes the same HeteroData/Batch as HeatGraphGNN, so SurrogateEvaluator, the
conformal calibration and NSGA-III use it unchanged. Fitted by least squares on the GNN's
training patches only (scripts/06_baselines.py protocol).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .graphs import EDGE_FEATURES

SHADE, OVER = EDGE_FEATURES.index("crown_shade_frac"), EDGE_FEATURES.index("overhead_frac")
REL = ("slot", "shades", "segment")


def _features(data):
    """(n_segments, 3) per-segment sums for a HeteroData or Batch."""
    n = data["segment"].num_nodes
    ei, ea = data[REL].edge_index, data[REL].edge_attr
    dst = ei[1]
    f = torch.zeros((n, 3), dtype=torch.float64)
    if dst.numel():
        f[:, 0].index_add_(0, dst, ea[:, SHADE].double())
        f[:, 1].index_add_(0, dst, ea[:, OVER].double())
        f[:, 2].index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float64))
    return f


class LinearPhysicsSurrogate:
    name = "linear_physics"

    def __init__(self, coef):
        self.coef = torch.as_tensor(np.asarray(coef, float))     # (hours, 3)

    # ---- fitting
    @classmethod
    def fit(cls, scene_graphs_and_labels) -> "LinearPhysicsSurrogate":
        """[(SceneGraph, plantings (n,K), utci (n,hours,E)), ...]; planting 0 = unplanted baseline."""
        A, y = {}, {}
        for sg, X, U in scene_graphs_and_labels:
            for i in range(1, len(X)):
                for h in range(sg.hours):
                    f = _features(sg.build(X[i], h)).numpy()
                    A.setdefault(h, []).append(f)
                    y.setdefault(h, []).append(U[i, h] - U[0, h])
        H = max(A) + 1
        coef = [np.linalg.lstsq(np.vstack(A[h]), np.concatenate(y[h]), rcond=None)[0] for h in range(H)]
        return cls(np.array(coef))

    # ---- GNN-compatible interface
    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter(())

    @torch.no_grad()
    def predict(self, data):
        f = _features(data)
        hour = data["segment"].hour.cpu()
        delta = (f * self.coef[hour]).sum(1)
        base = data["segment"].base.double().cpu() if "base" in data["segment"] else 0.0
        return (base + delta).float()

    # ---- persistence
    def save(self, path):
        Path(path).write_text(json.dumps({"model": self.name, "features": ["sum_crown_shade_frac",
                              "sum_overhead_frac", "n_planted_links"], "coef_per_hour": self.coef.tolist()},
                                         indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["coef_per_hour"])
