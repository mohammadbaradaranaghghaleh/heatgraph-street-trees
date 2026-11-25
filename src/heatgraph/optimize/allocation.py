"""Block H: NSGA-III street-tree allocation (pymoo).

Decision vector  x in {0,1}^K   (K candidate planting slots)
Objectives (all minimised)
  f1  person-weighted heat dose on the network (degC * person * h), computed from the
      conformal UPPER bound of surrogate UTCI -> robust to surrogate error (Block G)
  f2  inequality (Theil T by default, or Gini) of per-capita dose across neighbourhoods,
      population-weighted
  f3  number of trees (or cost if `slot_cost` is given)
Constraint       sum(cost * x) <= budget

Seeding and archive (guarantee against the greedy baseline)
  The initial population holds the whole greedy path (best 1, 2, ..., B trees), swap-mutated
  variants of each greedy design, the empty design and random feasible designs. Every design
  NSGA-III evaluates is stored in an archive, and the returned front is the non-dominated
  feasible subset of that archive. Because the greedy designs are in the archive, the front
  weakly dominates the greedy path: min dose <= greedy dose, and HV(front) >= HV(greedy).
  (NSGA-III's own survival step may drop a front-1 solution when front 1 exceeds the
  population size, so the archive, not the final population, carries the guarantee.)

`objectives_from_utci` is shared with Block I so surrogate and engine designs are scored
by exactly the same function.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from ..graph import equity
from ..graph.exposure import edge_heat_dose


def objectives_from_utci(utci, flows_hourly, lengths, vulnerability, groups, group_pop: dict,
                         threshold_c: float = 26.0, inequality: str = "theil"):
    """utci: (n, hours, segments) -> (dose_total (n,), inequality (n,), per-group per-capita (n, G))."""
    utci = np.asarray(utci, float)
    groups = np.asarray(groups)
    keys = sorted(group_pop)
    pops = np.array([group_pop[k] for k in keys], float)
    ineq_fn = equity.theil if inequality == "theil" else equity.gini
    tot, ineq, per_cap = [], [], []
    for u in utci:
        dose = edge_heat_dose(u, flows_hourly, lengths, vulnerability, threshold_c)
        pc = np.array([dose[groups == k].sum() for k in keys]) / pops
        tot.append(dose.sum()); ineq.append(ineq_fn(pc, pops)); per_cap.append(pc)
    return np.array(tot), np.array(ineq), np.array(per_cap)


class SurrogateEvaluator:
    """Scores plantings with the trained surrogate (+ conformal upper bound).

    `model` is a HeatGraphGNN or an Ensemble (anything with .predict(batch); an Ensemble
    also provides .predict_std(batch)). With a NormalizedConformal calibrator the upper
    bound is mean + q * std per segment; with SplitConformal it is mean + q.
    """

    def __init__(self, model, scene_graph, flows_hourly, lengths, vulnerability, groups, group_pop,
                 conformal=None, hour_groups=None, threshold_c: float = 26.0, inequality: str = "theil",
                 slot_cost=None):
        self.model, self.sg = model.eval(), scene_graph
        self.flows, self.lengths, self.vuln = flows_hourly, lengths, vulnerability
        self.groups, self.group_pop = groups, group_pop
        self.conformal, self.hour_groups = conformal, hour_groups
        self.threshold, self.inequality = threshold_c, inequality
        self.slot_cost = np.ones(scene_graph.scene.n_slots) if slot_cost is None else np.asarray(slot_cost, float)
        self.n_evals = 0
        params = list(model.parameters()) if hasattr(model, "parameters") else []
        self.device = params[0].device if params else torch.device("cpu")
        self.chunk = 32

    @torch.no_grad()
    def predict_utci(self, X, bound: str = "upper") -> np.ndarray:
        """bound: 'upper' | 'lower' | 'point'. Returns (n, hours, segments)."""
        X = np.atleast_2d(np.asarray(X))
        H, E = self.sg.hours, self.sg.scene.n_segments
        normalized = getattr(self.conformal, "normalized", False) and bound != "point"
        preds, stds = [], []
        for i in range(0, len(X), self.chunk):                  # bounded GPU memory
            xb = X[i:i + self.chunk]
            if hasattr(self.sg, "build_batch"):
                batch = self.sg.build_batch(xb)
            else:
                batch = Batch.from_data_list([self.sg.build(x, h) for x in xb for h in range(H)])
            batch = batch.to(self.device)
            preds.append(self.model.predict(batch).detach().cpu().numpy())
            if normalized:
                stds.append(self.model.predict_std(batch).detach().cpu().numpy())
        pred = np.concatenate(preds).reshape(len(X), H, E)
        self.n_evals += len(X)
        if bound == "point" or self.conformal is None:
            return pred
        sign = 1.0 if bound == "upper" else -1.0
        if normalized:
            std = np.concatenate(stds).reshape(len(X), H, E)
            return pred + sign * self.conformal.half_width_from_std(std)
        if self.hour_groups is not None and self.conformal.q_by_group is not None:
            q = np.array([self.conformal.half_width(np.array([g]))[0] for g in self.hour_groups])
            return pred + sign * q[None, :, None]
        return pred + sign * self.conformal.half_width()

    def objectives(self, X, bound: str = "upper") -> np.ndarray:
        X = np.atleast_2d(np.asarray(X))
        dose, ineq, _ = objectives_from_utci(self.predict_utci(X, bound), self.flows, self.lengths, self.vuln,
                                             self.groups, self.group_pop, self.threshold, self.inequality)
        return np.column_stack([dose, ineq, X.astype(float) @ self.slot_cost])

    def __call__(self, X, upper: bool = True) -> np.ndarray:
        return self.objectives(X, "upper" if upper else "point")


# --------------------------------------------------------------------------- baselines / seeds
def greedy_path(evaluator, budget: int) -> np.ndarray:
    """Greedy on f1: row b-1 is the greedy design with b trees (b = 1..budget)."""
    K = evaluator.sg.scene.n_slots
    x = np.zeros(K, int)
    path = []
    for _ in range(int(min(budget, K))):
        cand = [k for k in range(K) if x[k] == 0]
        trials = np.repeat(x[None], len(cand), 0)
        trials[np.arange(len(cand)), cand] = 1
        x = trials[int(np.argmin(evaluator(trials)[:, 0]))].copy()
        path.append(x.copy())
    return np.array(path, int).reshape(-1, K)


def greedy_allocation(evaluator, budget: int) -> np.ndarray:
    """Baseline: add, one at a time, the slot that most reduces f1 (dose)."""
    p = greedy_path(evaluator, budget)
    return p[-1] if len(p) else np.zeros(evaluator.sg.scene.n_slots, int)


def swap_mutants(x, n: int, rng) -> np.ndarray:
    """Variants of x with one planted slot moved to an unplanted slot (tree count preserved)."""
    x = np.asarray(x, int)
    on, off = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
    out = []
    for _ in range(n if len(on) and len(off) else 0):
        y = x.copy()
        y[rng.choice(on)] = 0
        y[rng.choice(off)] = 1
        out.append(y)
    return np.array(out, int).reshape(-1, len(x))


def seed_population(evaluator, budget: int, pop_size: int, mutants_per_seed: int = 2, seed: int = 0,
                    greedy=None) -> np.ndarray:
    """Greedy path + swap mutants + empty design + random feasible designs, deduplicated."""
    rng = np.random.default_rng(seed)
    K = evaluator.sg.scene.n_slots
    greedy = greedy_path(evaluator, budget) if greedy is None else np.asarray(greedy, int)
    parts = [greedy, np.zeros((1, K), int)] + [swap_mutants(g, mutants_per_seed, rng) for g in greedy]
    pop = np.unique(np.vstack(parts), axis=0)
    while len(pop) < pop_size:
        x = np.zeros(K, int)
        x[rng.choice(K, size=rng.integers(1, int(budget) + 1), replace=False)] = 1
        pop = np.unique(np.vstack([pop, x]), axis=0)
    if len(pop) > pop_size:   # keep all greedy designs + the empty design, trim the rest at random
        keep = {tuple(r) for r in np.vstack([greedy, np.zeros((1, K), int)])}
        must = np.array([r for r in pop if tuple(r) in keep])
        rest = np.array([r for r in pop if tuple(r) not in keep])
        rest = rest[rng.permutation(len(rest))[:max(pop_size - len(must), 0)]]
        pop = np.vstack([must, rest]) if len(rest) else must
    return pop.astype(bool)


# --------------------------------------------------------------------------- NSGA-III
def run_nsga3(evaluator, budget: float, pop_size: int = 40, n_gen: int = 25, n_partitions: int = 6,
              seed: int = 0, initial=None, seed_with_greedy: bool = True, greedy=None):
    """Return dict(X, F, G, archive_X, archive_F, greedy, n_evals) where (X, F) is the
    non-dominated feasible subset of every design evaluated (archive)."""
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    K = evaluator.sg.scene.n_slots
    cost = evaluator.slot_cost
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    pop_size = max(pop_size, len(ref_dirs))
    archive_X, archive_F, archive_G = [], [], []

    class TreeAllocation(Problem):
        def __init__(self):
            super().__init__(n_var=K, n_obj=3, n_ieq_constr=1, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *args, **kwargs):
            Xi = X.astype(int)
            F = evaluator(Xi)
            G = (Xi.astype(float) @ cost - budget)[:, None]
            archive_X.append(Xi); archive_F.append(F); archive_G.append(G)
            out["F"], out["G"] = F, G

    if initial is not None:
        sampling = np.asarray(initial, bool)
    elif seed_with_greedy:
        if greedy is None:
            greedy = greedy_path(evaluator, int(budget))
        sampling = seed_population(evaluator, int(budget), pop_size, seed=seed, greedy=greedy)
    else:
        sampling = BinaryRandomSampling()
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling,
                 crossover=TwoPointCrossover(), mutation=BitflipMutation(), eliminate_duplicates=True)
    minimize(TreeAllocation(), algo, ("n_gen", n_gen), seed=seed, verbose=False, save_history=False)

    AX, AF, AG = np.vstack(archive_X), np.vstack(archive_F), np.vstack(archive_G)
    AX, idx = np.unique(AX, axis=0, return_index=True)
    AF, AG = AF[idx], AG[idx]
    feas = AG[:, 0] <= 1e-9
    nd = np.zeros(len(AX), bool)
    nd[np.flatnonzero(feas)] = is_nondominated(AF[feas])
    return {"X": AX[nd], "F": AF[nd], "G": AG[nd], "archive_X": AX, "archive_F": AF,
            "greedy": greedy, "n_evals": evaluator.n_evals}


def is_nondominated(F) -> np.ndarray:
    F = np.asarray(F, float)
    nd = np.ones(len(F), bool)
    for i in range(len(F)):
        dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        nd[i] = not dominated_by.any()
    return nd


def hypervolume(F, ref_point) -> float:
    """Hypervolume of a minimisation front w.r.t. ref_point (pymoo HV indicator)."""
    from pymoo.indicators.hv import HV

    F = np.atleast_2d(np.asarray(F, float))
    F = F[np.all(F < np.asarray(ref_point), axis=1)]
    return float(HV(ref_point=np.asarray(ref_point, float))(F)) if len(F) else 0.0


def compare_fronts(fronts: dict, ref_point=None) -> dict:
    """Hypervolume of several fronts against a shared reference point (default: 1.1 x the
    worst value of each objective over all fronts). Returns {name: hv, ..., 'ref_point': rp}."""
    allF = np.vstack([np.atleast_2d(F) for F in fronts.values()])
    rp = np.asarray(ref_point, float) if ref_point is not None else allF.max(0) * 1.1 + 1e-9
    out = {name: hypervolume(F, rp) for name, F in fronts.items()}
    out["ref_point"] = rp
    return out
