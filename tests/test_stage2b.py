"""Stage 2b: ensemble + normalised conformal, greedy-seeded NSGA-III with hypervolume,
and the Block I active-learning loop. CPU only, synthetic scene."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "torch", "torch_geometric", "pymoo", "pvlib",
                                                       "pythermalcomfort", "networkx"))

if HAVE_DEPS:
    import numpy as np
    import torch
    from torch_geometric.data import Batch

    from heatgraph.optimize.allocation import (SurrogateEvaluator, compare_fronts, greedy_path, is_nondominated,
                                               run_nsga3, seed_population)
    from heatgraph.surrogate.ensemble import Ensemble
    from heatgraph.surrogate.graphs import SceneGraph
    from heatgraph.synthetic import generate_labels, random_plantings, toy_scene
    from heatgraph.uq.conformal import NormalizedConformal, SplitConformal
    from heatgraph.validate.active_learning import active_learning_loop, calibrate
    from heatgraph.validate.pareto_gap import surrogate_optimality_gap
    from test_pipeline_e2e import segment_flows

BUDGET = 6


@unittest.skipUnless(HAVE_DEPS, "ML stack not installed (use HeatGraph/.venv)")
class TestStage2b(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.s = s = toy_scene(seed=0)
        X = random_plantings(s, 45, p=0.35, seed=7)
        Y = generate_labels(s, X)
        cls.X, cls.Y = X, Y
        cls.sg = SceneGraph(s, baseline_utci=Y[0])
        cls.tr = slice(0, 20); cls.cal = slice(20, 35); cls.te = slice(35, 45)
        cls.ens = Ensemble.train(cls.sg.dataset(X[cls.tr], Y[cls.tr]), n_members=3, epochs=40,
                                 model_kwargs={"hidden": 32})
        cls.cp = calibrate(cls.ens, cls.sg, X[cls.cal], Y[cls.cal], alpha=0.1)
        cls.flows, cls.lengths = segment_flows(s)

    def make_ev(self, model=None, cp=None):
        s = self.s
        return SurrogateEvaluator(model or self.ens, self.sg, self.flows, self.lengths, np.ones(s.n_segments),
                                  s.group, s.population, conformal=cp or self.cp)

    # ---- Block G (normalised conformal)
    def test_normalized_conformal_synthetic(self):
        rng = np.random.default_rng(1)
        sd_c, sd_t = rng.uniform(0.1, 2.0, 3000), rng.uniform(0.1, 2.0, 6000)
        yh_c, yh_t = rng.normal(size=3000), rng.normal(size=6000)
        y_c, y_t = yh_c + rng.normal(size=3000) * sd_c, yh_t + rng.normal(size=6000) * sd_t
        nc = NormalizedConformal(0.1).calibrate(y_c, yh_c, sd_c)
        lo, hi = nc.interval(yh_t, sd_t)
        self.assertTrue(0.88 <= NormalizedConformal.coverage(y_t, lo, hi) <= 0.92)
        self.assertGreater(np.corrcoef(hi - lo, sd_t)[0, 1], 0.99)      # widths adapt to difficulty
        # in the noisiest third, adaptive intervals keep coverage where constant ones fail
        sc = SplitConformal(0.1).calibrate(y_c, yh_c)
        hard = sd_t > np.quantile(sd_t, 2 / 3)
        cov_split = SplitConformal.coverage(y_t[hard], *sc.interval(yh_t[hard]))
        cov_norm = NormalizedConformal.coverage(y_t[hard], lo[hard], hi[hard])
        self.assertGreater(cov_norm, cov_split)

    def test_ensemble_conformal_coverage_on_unseen_plantings(self):
        data = self.sg.dataset(self.X[self.te], self.Y[self.te])
        b = Batch.from_data_list(data)
        with torch.no_grad():
            mu, sd = self.ens.predict(b).numpy(), self.ens.predict_std(b).numpy()
        lo, hi = self.cp.interval(mu, sd)
        self.assertGreaterEqual(NormalizedConformal.coverage(b["segment"].y.numpy(), lo, hi), 0.8)
        self.assertGreater(np.std(hi - lo), 0)                           # design-dependent widths

    # ---- Block H (seeded NSGA-III)
    def test_seed_population_contents(self):
        ev = self.make_ev()
        g = greedy_path(ev, BUDGET)
        self.assertEqual(g.shape, (BUDGET, self.s.n_slots))
        self.assertTrue(np.all(g.sum(1) == np.arange(1, BUDGET + 1)))
        pop = seed_population(ev, BUDGET, 30, greedy=g).astype(int)
        self.assertEqual(len(pop), 30)
        self.assertEqual(len(np.unique(pop, axis=0)), 30)
        self.assertTrue(np.all(pop.sum(1) <= BUDGET))
        rows = {tuple(r) for r in pop}
        self.assertTrue(all(tuple(r) in rows for r in g))
        self.assertIn(tuple(np.zeros(self.s.n_slots, int)), rows)

    def test_seeded_nsga3_dominates_greedy_and_logs_hypervolume(self):
        ev = self.make_ev()
        g = greedy_path(ev, BUDGET)
        Fg = ev(g)
        seeded = run_nsga3(ev, budget=BUDGET, pop_size=30, n_gen=8, seed=0, greedy=g)
        plain = run_nsga3(ev, budget=BUDGET, pop_size=30, n_gen=8, seed=0, seed_with_greedy=False)
        F = seeded["F"]
        self.assertTrue(is_nondominated(F).all())
        self.assertTrue(np.all(seeded["X"].sum(1) <= BUDGET))
        # The same design can be scored in different batches (float32 summation order), so
        # objective values agree to ~1e-6 relative, not bit-for-bit.
        tol = lambda v: 1e-5 * np.abs(v) + 1e-9
        self.assertLessEqual(F[:, 0].min(), Fg[:, 0].min() + tol(Fg[:, 0].min()))   # matches or beats greedy dose
        for fg in Fg:                                                    # weakly dominates every greedy design
            self.assertTrue(np.any(np.all(F <= fg + tol(fg), axis=1)))
        hv = compare_fronts({"seeded": F, "unseeded": plain["F"], "greedy": Fg[is_nondominated(Fg)]})
        self.assertGreaterEqual(hv["seeded"], hv["greedy"] - 1e-12)
        self.assertGreater(hv["seeded"], 0)
        print(f"\n  HV seeded={hv['seeded']:.4g} unseeded={hv['unseeded']:.4g} greedy={hv['greedy']:.4g} "
              f"| min dose seeded={F[:, 0].min():.2f} unseeded={plain['F'][:, 0].min():.2f} "
              f"greedy={Fg[:, 0].min():.2f}")

    # ---- Block I (active learning)
    def _engine(self, X):
        return generate_labels(self.s, X)

    def test_active_learning_rounds(self):
        import copy
        ens = Ensemble([copy.deepcopy(m) for m in self.ens.members])   # refit mutates members; keep class copy clean
        X0, Y0 = self.X[self.tr], self.Y[self.tr]
        cand = lambda ev, r: run_nsga3(ev, budget=BUDGET, pop_size=30, n_gen=3, seed=r)["archive_X"]
        out = active_learning_loop(self.sg, ens, X0, Y0, self.X[self.cal], self.Y[self.cal],
                                   lambda m, cp: self.make_ev(m, cp), self._engine, budget=BUDGET, rounds=2,
                                   per_round=3, gap_threshold=0.0, refit_epochs=15, candidate_fn=cand)
        h = out["history"]
        self.assertEqual(len(h), 2)
        known = {tuple(x) for x in np.vstack([X0, self.X[self.cal]])}
        for rec in h:
            self.assertEqual(len(rec["selected"]), 3)
            self.assertTrue(all(tuple(x) not in known for x in rec["selected"]))   # genuinely new designs
            self.assertTrue(np.all(rec["width_selected"] >= rec["width_pool_median"] - 1e-9))
            self.assertTrue(np.all(np.isfinite(rec["rel_gap"])))
        self.assertEqual(len(out["train_X"]), len(X0) + 6)
        # after refit, the designs acquired in round 0 are predicted at least as well as before
        sel0 = h[0]["selected"]
        cp = calibrate(ens, self.sg, self.X[self.cal], self.Y[self.cal], 0.1)
        after = surrogate_optimality_gap(self.make_ev(ens, cp), sel0, self._engine(sel0))["rel_gap"]
        self.assertLessEqual(np.mean(np.abs(after)), np.mean(np.abs(h[0]["rel_gap"])) + 0.005)

    def test_active_learning_stops_when_gap_small(self):
        import copy
        ens = Ensemble([copy.deepcopy(m) for m in self.ens.members])
        cand = lambda ev, r: run_nsga3(ev, budget=BUDGET, pop_size=30, n_gen=2, seed=r)["archive_X"]
        out = active_learning_loop(self.sg, ens, self.X[self.tr], self.Y[self.tr], self.X[self.cal],
                                   self.Y[self.cal], lambda m, cp: self.make_ev(m, cp), self._engine,
                                   budget=BUDGET, rounds=3, per_round=2, gap_threshold=1.0, candidate_fn=cand)
        self.assertTrue(out["converged"])
        self.assertEqual(len(out["history"]), 1)
        self.assertEqual(len(out["train_X"]), 20)                 # nothing added once converged


if __name__ == "__main__":
    unittest.main()
