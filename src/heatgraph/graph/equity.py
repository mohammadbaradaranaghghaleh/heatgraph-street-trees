"""Inequality metrics for the distribution of heat dose across neighbourhoods.

All functions accept optional population weights: each neighbourhood i has a per-capita
value x_i and a population w_i. Values must be non-negative (Gini) / positive (Theil:
zeros are handled with the 0*ln(0) = 0 convention).

  gini(x, w)            weighted Gini via the Lorenz curve (0 = equal, -> 1 = concentrated)
  theil(x, w)           Theil T index, 0 = equal, max ln(N) for N equal-weight units
  theil_decomposition   T = T_between + T_within for a grouping (e.g. districts, income bands)
"""
from __future__ import annotations

import numpy as np


def _prep(x, w):
    x = np.asarray(x, dtype=float).ravel()
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float).ravel()
    if x.shape != w.shape:
        raise ValueError("x and w must have the same length")
    if np.any(x < 0) or np.any(w < 0) or not np.all(np.isfinite(x)):
        raise ValueError("values must be finite and non-negative; weights non-negative")
    keep = w > 0
    return x[keep], w[keep]


def gini(x, w=None) -> float:
    x, w = _prep(x, w)
    if x.size == 0 or np.sum(x * w) == 0:
        return 0.0
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) / w.sum()                       # population share
    cx = np.cumsum(x * w) / np.sum(x * w)             # dose share (Lorenz curve)
    cw0, cx0 = np.concatenate([[0], cw]), np.concatenate([[0], cx])
    area = np.sum((cw0[1:] - cw0[:-1]) * (cx0[1:] + cx0[:-1]) / 2.0)   # trapezoids under Lorenz
    return float(1.0 - 2.0 * area)


def theil(x, w=None) -> float:
    x, w = _prep(x, w)
    if x.size == 0:
        return 0.0
    mu = np.sum(w * x) / w.sum()
    if mu == 0:
        return 0.0
    r = x / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(r > 0, r * np.log(r), 0.0)
    return float(np.sum(w * terms) / w.sum())


def theil_decomposition(x, groups, w=None) -> dict:
    """Return dict(total, between, within, by_group) with total == between + within."""
    x = np.asarray(x, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float).ravel()
    total_w, mu = w.sum(), np.sum(w * x) / w.sum()
    between, within, by_group = 0.0, 0.0, {}
    for g in np.unique(groups):
        m = groups == g
        wg, xg = w[m], x[m]
        share_pop = wg.sum() / total_w
        mu_g = np.sum(wg * xg) / wg.sum()
        share_dose = share_pop * mu_g / mu if mu > 0 else 0.0
        t_g = theil(xg, wg)
        by_group[g] = {"mean": mu_g, "theil": t_g, "pop_share": share_pop}
        if mu_g > 0:
            between += share_dose * np.log(mu_g / mu)
        within += share_dose * t_g
    return {"total": theil(x, w), "between": float(between), "within": float(within), "by_group": by_group}
