"""Block I (part): surrogate-optimality gap.

Re-simulates optimised plantings with the physics engine and compares the objective the
optimiser believed (surrogate point prediction and conformal upper bound) with the objective
the engine produces. Reported per design:
  rel_gap        (dose_surrogate - dose_engine) / dose_engine      (point prediction)
  upper_covers   dose_upper >= dose_engine                           (robust objective held)
"""
from __future__ import annotations

import numpy as np

from ..optimize.allocation import objectives_from_utci


def surrogate_optimality_gap(evaluator, X, engine_utci) -> dict:
    """engine_utci: (n, hours, segments) from the engine for the same plantings X."""
    X = np.atleast_2d(X)
    args = (evaluator.flows, evaluator.lengths, evaluator.vuln, evaluator.groups, evaluator.group_pop,
            evaluator.threshold, evaluator.inequality)
    d_eng, i_eng, _ = objectives_from_utci(engine_utci, *args)
    d_pt, i_pt, _ = objectives_from_utci(evaluator.predict_utci(X, "point"), *args)
    d_up, _, _ = objectives_from_utci(evaluator.predict_utci(X, "upper"), *args)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(d_eng > 0, (d_pt - d_eng) / d_eng, 0.0)
    return {"dose_engine": d_eng, "dose_surrogate": d_pt, "dose_upper": d_up, "rel_gap": rel,
            "upper_covers": d_up >= d_eng - 1e-9, "ineq_engine": i_eng, "ineq_surrogate": i_pt}
