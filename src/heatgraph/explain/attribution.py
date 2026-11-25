"""Block F: attributions for the graph surrogate (no captum dependency).

integrated_gradients   Sundararajan et al. (2017) path integral, per node and feature, for any
                       node type of a HeteroData; satisfies completeness
                       (sum of attributions = f(x) - f(baseline)) up to the Riemann error,
                       which is returned so it can be reported.
slot_contributions     counterfactual leave-one-out: change in target when each planted slot
                       is removed (exact, model-based "which tree does the work").
permutation_importance model-agnostic feature importance (MAE increase when a feature column
                       is shuffled across samples), for global rankings by feature group.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..surrogate.train import evaluate


def _target(model, data, x_dict, segment_mask, edge_attr_dict=None, base=None):
    pred = model(x_dict, data.edge_index_dict, edge_attr_dict if edge_attr_dict is not None else data.edge_attr_dict,
                 base)
    return pred[segment_mask].sum() if segment_mask is not None else pred.sum()


def integrated_gradients(model, data, node_type: str = "segment", segment_mask=None, baseline=None,
                         steps: int = 64) -> dict:
    """Return dict(attr=(n_nodes, n_features) array, f_x, f_baseline, completeness_error).

    The target is the summed predicted UTCI of `segment_mask` segments (all if None).
    Default baseline: zeros for the chosen node type (other node types kept at their values).
    Runs on a float64 copy of the model: the target is a sum of ~10^2 degC values, where
    float32 rounding alone (~1e-2 degC) would mask the completeness check.
    """
    model = copy.deepcopy(model).double().eval()
    x0 = data[node_type].x.detach().double()
    base = torch.zeros_like(x0) if baseline is None else torch.as_tensor(baseline, dtype=torch.float64)
    xs = {k: v.detach().double() for k, v in data.x_dict.items()}
    ea = {k: v.double() for k, v in data.edge_attr_dict.items()}
    seg_base = data["segment"].base.double() if "base" in data["segment"] else None
    _t = lambda xd: _target(model, data, xd, segment_mask, ea, seg_base)
    grads = torch.zeros_like(x0)
    for a in (torch.arange(steps, dtype=x0.dtype) + 0.5) / steps:      # midpoint Riemann rule
        xi = (base + a * (x0 - base)).requires_grad_(True)
        g, = torch.autograd.grad(_t({**xs, node_type: xi}), xi)
        grads += g
    attr = (x0 - base) * grads / steps
    with torch.no_grad():
        f_x = float(_t(xs))
        f_b = float(_t({**xs, node_type: base}))
    return {"attr": attr.numpy(), "f_x": f_x, "f_baseline": f_b,
            "completeness_error": float(attr.sum()) - (f_x - f_b)}


@torch.no_grad()
def slot_contributions(model, scene_graph, planting, hour: int, segment_mask=None) -> np.ndarray:
    """For each planted slot k: target(planting) - target(planting without k) (degC summed)."""
    model.eval()
    x = np.asarray(planting).copy()

    def f(p):
        d = scene_graph.build(p, hour)
        pred = model.predict(d)
        return float(pred[segment_mask].sum() if segment_mask is not None else pred.sum())

    full = f(x)
    out = np.zeros(len(x))
    for k in np.flatnonzero(x):
        x[k] = 0
        out[k] = full - f(x)
        x[k] = 1
    return out


def permutation_importance(model, data_list: list, feature_names: list, node_type: str = "segment",
                           n_repeats: int = 3, seed: int = 0) -> dict:
    """MAE increase (degC) when one feature column is permuted across all nodes of all samples."""
    rng = np.random.default_rng(seed)
    base_mae = evaluate(model, data_list)["mae"]
    stacked = torch.cat([d[node_type].x for d in data_list])
    sizes = [d[node_type].x.shape[0] for d in data_list]
    out = {}
    for j, name in enumerate(feature_names):
        inc = []
        for _ in range(n_repeats):
            perm = stacked.clone()
            perm[:, j] = stacked[rng.permutation(len(stacked)), j]
            parts = torch.split(perm, sizes)
            shuffled = []
            for d, xnew in zip(data_list, parts):
                d2 = copy.copy(d)
                d2[node_type].x = xnew
                shuffled.append(d2)
            inc.append(evaluate(model, shuffled)["mae"] - base_mae)
        out[name] = float(np.mean(inc))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
