"""Block G: split-conformal prediction intervals for segment UTCI.

Given calibration residuals r_i = |y_i - yhat_i| (n of them), the half-width
    q = the ceil((n + 1)(1 - alpha))-th smallest residual
gives marginal coverage P(y in [yhat - q, yhat + q]) >= 1 - alpha for exchangeable data
(Angelopoulos & Bates, 2021). `Mondrian` mode calibrates one q per group (e.g. per hour of
day), which keeps coverage within each group. Calibration data must come from plantings
that were used neither for training nor for model selection.

The optimiser uses the upper bound yhat + q as a robust, uncertainty-aware objective.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(residuals, alpha: float) -> float:
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)).ravel())
    n = r.size
    if n == 0:
        raise ValueError("no calibration residuals")
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(r[k - 1])


class SplitConformal:
    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.q = None
        self.q_by_group = None

    def calibrate(self, y_true, y_pred, groups=None):
        res = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
        self.q = conformal_quantile(res, self.alpha)
        if groups is not None:
            g = np.asarray(groups)
            self.q_by_group = {k: conformal_quantile(res[g == k], self.alpha) for k in np.unique(g)}
        self.n_cal = res.size
        return self

    def half_width(self, groups=None):
        if self.q is None:
            raise RuntimeError("calibrate() first")
        if groups is None or self.q_by_group is None:
            return self.q
        return np.vectorize(lambda k: self.q_by_group.get(k, self.q))(np.asarray(groups))

    def interval(self, y_pred, groups=None):
        w = self.half_width(groups)
        y = np.asarray(y_pred, float)
        return y - w, y + w

    def upper(self, y_pred, groups=None):
        return np.asarray(y_pred, float) + self.half_width(groups)

    @staticmethod
    def coverage(y_true, lo, hi) -> float:
        y = np.asarray(y_true, float)
        return float(np.mean((y >= lo) & (y <= hi)))


class NormalizedConformal:
    """Locally adaptive split conformal (normalised residual scores).

    score_i = |y_i - yhat_i| / (std_i + eps); interval = yhat +/- q * (std + eps).
    `std` is a difficulty estimate (here: deep-ensemble spread). Coverage stays marginal
    >= 1 - alpha for exchangeable data; widths now vary by design and segment, which is
    what the active-learning acquisition needs (plain split conformal is constant width).
    """

    normalized = True

    def __init__(self, alpha: float = 0.1, eps: float = 0.02):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha, self.eps, self.q, self.q_by_group = alpha, eps, None, None

    def calibrate(self, y_true, y_pred, std):
        s = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) / (np.asarray(std, float) + self.eps)
        self.q = conformal_quantile(s, self.alpha)
        self.n_cal = s.size
        return self

    def half_width_from_std(self, std):
        if self.q is None:
            raise RuntimeError("calibrate() first")
        return self.q * (np.asarray(std, float) + self.eps)

    def interval(self, y_pred, std):
        w = self.half_width_from_std(std)
        y = np.asarray(y_pred, float)
        return y - w, y + w

    coverage = staticmethod(SplitConformal.coverage)
