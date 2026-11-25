"""Exact fast evaluator for the linear physics surrogate (additive in planted slots).

UTCI(x)[h, e] = base[h, e] + sum_k x_k * C[k, h, e], with C[k, h, e] the linear model's
contribution of slot k (sum over its links to segment e of coef_h . [shade, overhead, 1]).
Identical to LinearPhysicsSurrogate.predict on the graph (asserted in tests), but one
matrix product per planting, so greedy over hundreds of candidates is instant.
Same interface as SurrogateEvaluator (callable, predict_utci, sg, slot_cost, n_evals, ...).
"""
from __future__ import annotations

import numpy as np

from ..surrogate.linear import OVER, SHADE
from .allocation import objectives_from_utci


class AdditiveEvaluator:
    def __init__(self, scene_graph, coef, flows_hourly, lengths, vulnerability, groups, group_pop, conformal=None,
                 hour_groups=None, threshold_c: float = 26.0, inequality: str = "theil", slot_cost=None):
        self.sg = scene_graph
        sg = scene_graph
        K, H, E = sg.scene.n_slots, sg.hours, sg.scene.n_segments
        coef = np.asarray(coef, float)
        C = np.zeros((K, H, E))
        for h in range(H):
            a = sg.link_attr[h]
            contrib = coef[h, 0] * a[:, SHADE] + coef[h, 1] * a[:, OVER] + coef[h, 2]
            np.add.at(C[:, h, :], (sg.link_slot, sg.link_seg), contrib)
        self.C = C.reshape(K, -1)
        self.base = np.asarray(sg.baseline, float)
        self.flows, self.lengths, self.vuln = flows_hourly, lengths, vulnerability
        self.groups, self.group_pop = groups, group_pop
        self.conformal, self.hour_groups = conformal, hour_groups
        self.threshold, self.inequality = threshold_c, inequality
        self.slot_cost = np.ones(K) if slot_cost is None else np.asarray(slot_cost, float)
        self.n_evals = 0

    def eval(self):
        return self

    def _q(self):
        H = self.sg.hours
        if self.conformal is None:
            return np.zeros(H)
        if self.hour_groups is not None and getattr(self.conformal, "q_by_group", None):
            return np.array([self.conformal.half_width(np.array([g]))[0] for g in self.hour_groups])
        return np.full(H, self.conformal.half_width())

    def predict_utci(self, X, bound: str = "upper") -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        H, E = self.sg.hours, self.sg.scene.n_segments
        u = self.base[None] + (X @ self.C).reshape(len(X), H, E)
        self.n_evals += len(X)
        if bound == "point":
            return u
        return u + (1 if bound == "upper" else -1) * self._q()[None, :, None]

    def objectives(self, X, bound: str = "upper") -> np.ndarray:
        X = np.atleast_2d(np.asarray(X))
        dose, ineq, _ = objectives_from_utci(self.predict_utci(X, bound), self.flows, self.lengths, self.vuln,
                                             self.groups, self.group_pop, self.threshold, self.inequality)
        return np.column_stack([dose, ineq, X.astype(float) @ self.slot_cost])

    def __call__(self, X, upper: bool = True) -> np.ndarray:
        return self.objectives(X, "upper" if upper else "point")
