"""Block I: active-learning loop around the optimiser.

Each round
  1. calibrate NormalizedConformal on a FIXED calibration set (ensemble mean / spread);
  2. build the evaluator and generate a candidate pool (default: every feasible design
     evaluated by a seeded NSGA-III run, i.e. the designs the optimiser actually visits);
  3. acquisition = width of the conformal dose interval (dose_upper - dose_lower); the
     `per_round` widest designs not yet simulated are selected;
  4. the physics engine re-simulates them; the surrogate-optimality gap is recorded
     BEFORE retraining (an honest estimate of the gap the optimiser was facing);
  5. stop if max |relative dose gap| <= gap_threshold, otherwise add the designs to the
     training set and warm-start the ensemble.

Caveat for the paper: acquired designs are chosen by the model, so they are not
exchangeable with the calibration set; coverage on them is reported, not guaranteed.
The calibration set is never used for training.
"""
from __future__ import annotations

import numpy as np

from ..optimize.allocation import run_nsga3
from ..uq.conformal import NormalizedConformal
from .pareto_gap import surrogate_optimality_gap


def default_candidates(ev, budget: int, n_gen: int = 6, pop_size: int = 30, seed: int = 0):
    res = run_nsga3(ev, budget=budget, pop_size=pop_size, n_gen=n_gen, seed=seed)
    ax, af = res["archive_X"], res["archive_F"]
    return ax[ax @ ev.slot_cost <= budget + 1e-9]


def calibrate(ensemble, scene_graph, cal_X, cal_Y, alpha: float):
    import torch
    from torch_geometric.data import Batch

    data = scene_graph.dataset(cal_X, cal_Y)
    batch = Batch.from_data_list(data)
    with torch.no_grad():
        mu, sd = ensemble.predict(batch).numpy(), ensemble.predict_std(batch).numpy()
    y = batch["segment"].y.numpy()
    return NormalizedConformal(alpha).calibrate(y, mu, sd)


def active_learning_loop(scene_graph, ensemble, train_X, train_Y, cal_X, cal_Y, make_evaluator, engine_fn,
                         budget: int, rounds: int = 3, per_round: int = 4, gap_threshold: float = 0.02,
                         alpha: float = 0.1, refit_epochs: int = 30, candidate_fn=None, seed: int = 0) -> dict:
    """make_evaluator(model, conformal) -> SurrogateEvaluator; engine_fn(X) -> (n, hours, segments)."""
    train_X, train_Y = np.asarray(train_X), np.asarray(train_Y)
    seen = {tuple(x) for x in train_X} | {tuple(x) for x in np.asarray(cal_X)}
    candidate_fn = candidate_fn or (lambda ev, r: default_candidates(ev, budget, seed=seed + r))
    history, converged = [], False
    for r in range(rounds):
        cp = calibrate(ensemble, scene_graph, cal_X, cal_Y, alpha)
        ev = make_evaluator(ensemble, cp)
        pool = np.array([x for x in np.unique(candidate_fn(ev, r), axis=0) if tuple(x) not in seen])
        if len(pool) == 0:
            history.append({"round": r, "pool": 0, "stopped": "empty pool"})
            break
        width = ev.objectives(pool, "upper")[:, 0] - ev.objectives(pool, "lower")[:, 0]
        pick = np.argsort(-width)[:per_round]
        sel = pool[pick]
        Y_new = engine_fn(sel)
        gap = surrogate_optimality_gap(ev, sel, Y_new)
        rec = {"round": r, "pool": len(pool), "q": cp.q, "selected": sel, "width_selected": width[pick],
               "width_pool_median": float(np.median(width)), "rel_gap": gap["rel_gap"],
               "max_abs_gap": float(np.max(np.abs(gap["rel_gap"]))), "upper_covers": gap["upper_covers"],
               "n_train_before": len(train_X)}
        history.append(rec)
        seen |= {tuple(x) for x in sel}
        if rec["max_abs_gap"] <= gap_threshold:
            rec["stopped"], converged = "gap below threshold", True
            break
        train_X, train_Y = np.vstack([train_X, sel]), np.concatenate([train_Y, Y_new])
        ensemble.refit(scene_graph.dataset(train_X, train_Y), epochs=refit_epochs, seed=seed + r)
    return {"history": history, "train_X": train_X, "train_Y": train_Y, "ensemble": ensemble,
            "converged": converged}
