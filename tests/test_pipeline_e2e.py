"""End-to-end synthetic pipeline (Blocks C -> E -> F -> G -> H -> I), CPU only.

toy street network -> engine labels for random counterfactual plantings -> GNN surrogate
trained for a few epochs on CPU -> attributions -> split-conformal intervals ->
NSGA-III allocation -> engine re-simulation of the Pareto set.
Runtime ~30-60 s on a laptop CPU. Needs HeatGraph/.venv (torch CPU build, pyg, pymoo, mapie).
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # belt and braces: never touch a GPU here
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in
                ("numpy", "torch", "torch_geometric", "pymoo", "mapie", "sklearn", "pvlib", "pythermalcomfort"))

if HAVE_DEPS:
    import networkx as nx
    import numpy as np
    import torch

    from heatgraph.explain.attribution import integrated_gradients, permutation_importance, slot_contributions
    from heatgraph.graph.flows import gravity_flows
    from heatgraph.optimize.allocation import SurrogateEvaluator, greedy_allocation, is_nondominated, run_nsga3
    from heatgraph.surrogate.graphs import SEG_HOURLY, SEG_STATIC, SceneGraph
    from heatgraph.surrogate.model import HeatGraphGNN
    from heatgraph.surrogate.train import evaluate, fit, predict
    from heatgraph.synthetic import generate_labels, random_plantings, toy_scene
    from heatgraph.uq.conformal import SplitConformal
    from heatgraph.validate.pareto_gap import surrogate_optimality_gap

N_TRAIN, N_VAL, N_CAL, N_TEST = 40, 10, 10, 10


def segment_flows(scene, hourly_profile=(0.6, 1.0, 0.8)):
    G = nx.Graph()
    for u, v in scene.edges:
        (r0, c0), (r1, c1) = scene.nodes[u], scene.nodes[v]
        G.add_edge(u, v, length=float(abs(r1 - r0) + abs(c1 - c0)))
    centre = (1, 1)
    dest = {n: (3.0 if n == centre else 1.0) for n in G.nodes}
    f = gravity_flows(G, {n: 100.0 for n in G.nodes}, dest, radius_m=200)
    key = lambda u, v: (u, v) if str(u) <= str(v) else (v, u)
    per_seg = np.array([f[key(u, v)] for u, v in scene.edges])
    lengths = np.array([G[u][v]["length"] for u, v in scene.edges])
    return np.outer(hourly_profile, per_seg), lengths


@unittest.skipUnless(HAVE_DEPS, "ML stack not installed (use HeatGraph/.venv)")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.scene = s = toy_scene(seed=0)
        n = N_TRAIN + N_VAL + N_CAL + N_TEST
        cls.X = random_plantings(s, n, p=0.35, seed=1)
        cls.Y = generate_labels(s, cls.X)                       # engine labels (n, hours, segments)
        cls.sg = SceneGraph(s, baseline_utci=cls.Y[0])
        sl = np.cumsum([0, N_TRAIN, N_VAL, N_CAL, N_TEST])
        cls.idx = {k: slice(sl[i], sl[i + 1]) for i, k in enumerate(["train", "val", "cal", "test"])}
        cls.data = {k: cls.sg.dataset(cls.X[v], cls.Y[v]) for k, v in cls.idx.items()}
        cls.model = HeatGraphGNN(hidden=48)
        cls.hist = fit(cls.model, cls.data["train"], cls.data["val"], epochs=80, lr=3e-3, device="cpu")
        cls.flows, cls.lengths = segment_flows(s)

    # ---- Block E
    def test_cpu_only(self):
        self.assertTrue(all(p.device.type == "cpu" for p in self.model.parameters()))

    def test_surrogate_skill_on_unseen_plantings(self):
        r = evaluate(self.model, self.data["test"])
        y_train_mean = np.concatenate([d["segment"].y.numpy() for d in self.data["train"]]).mean()
        self.assertLess(r["mae"], 0.3)
        self.assertLess(r["mae"], 0.25 * np.mean(np.abs(r["true"] - y_train_mean)))
        # skill on the tree-induced CHANGE, which is what the optimiser exploits
        n_test = self.idx["test"].stop - self.idx["test"].start
        d_true = (self.Y[self.idx["test"]] - self.Y[0]).ravel()
        d_pred = (r["pred"].reshape(n_test, self.sg.hours, -1) - self.Y[0]).ravel()
        self.assertGreater(np.corrcoef(d_true, d_pred)[0, 1], 0.9)
        self.assertLess(np.mean(np.abs(d_true - d_pred)), 0.5 * np.mean(np.abs(d_true)))

    def test_more_trees_cool_at_noon(self):
        none, full = np.zeros(self.scene.n_slots, int), np.ones(self.scene.n_slots, int)
        p = predict(self.model, [self.sg.build(none, 1), self.sg.build(full, 1)]).reshape(2, -1)
        self.assertLess(p[1].mean(), p[0].mean() - 0.3)

    # ---- Block F
    def test_integrated_gradients_completeness(self):
        x = self.X[self.idx["test"]][0]
        d = self.sg.build(x, 1)
        # Completeness: sum(attr) = f(x) - f(baseline) up to the Riemann error. Positive and
        # negative attributions largely cancel, so the error is judged against the total
        # attribution mass and must shrink as the path is refined.
        for node_type in ("slot", "segment"):
            coarse = integrated_gradients(self.model, d, node_type=node_type, steps=32)
            fine = integrated_gradients(self.model, d, node_type=node_type, steps=512)
            mass = np.abs(fine["attr"]).sum()
            self.assertLess(abs(fine["completeness_error"]), 0.005 * mass + 1e-6, node_type)
            self.assertLessEqual(abs(fine["completeness_error"]), abs(coarse["completeness_error"]) + 1e-9, node_type)
        ig = integrated_gradients(self.model, d, node_type="slot", steps=64)
        unplanted = np.flatnonzero(x == 0)
        self.assertTrue(np.allclose(ig["attr"][unplanted], 0.0))   # no edges -> no influence

    def test_slot_contributions_and_permutation(self):
        x = np.ones(self.scene.n_slots, int)
        c = slot_contributions(self.model, self.sg, x, hour=1)
        self.assertLess(c.sum(), 0)                                 # trees cool in aggregate
        imp = permutation_importance(self.model, self.data["test"][:9], SEG_STATIC + SEG_HOURLY, n_repeats=1)
        self.assertEqual(len(imp), len(SEG_STATIC) + len(SEG_HOURLY))
        self.assertGreater(max(imp.values()), 0.0)

    # ---- Block G
    def _cal(self, alpha=0.1, mondrian=False):
        cal, test = self.data["cal"], self.data["test"]
        yc = np.concatenate([d["segment"].y.numpy() for d in cal])
        hours_c = np.concatenate([np.full(d["segment"].y.numel(), i % self.sg.hours) for i, d in enumerate(cal)])
        cp = SplitConformal(alpha).calibrate(yc, predict(self.model, cal), hours_c if mondrian else None)
        return cp, test

    def test_conformal_coverage_on_unseen_plantings(self):
        for mondrian in (False, True):
            cp, test = self._cal(0.1, mondrian)
            yt = np.concatenate([d["segment"].y.numpy() for d in test])
            ht = np.concatenate([np.full(d["segment"].y.numel(), i % self.sg.hours) for i, d in enumerate(test)])
            lo, hi = cp.interval(predict(self.model, test), ht if mondrian else None)
            self.assertGreaterEqual(SplitConformal.coverage(yt, lo, hi), 0.8, f"mondrian={mondrian}")

    def test_conformal_matches_theory_and_mapie(self):
        from mapie.regression import SplitConformalRegressor
        from sklearn.base import BaseEstimator, RegressorMixin

        rng = np.random.default_rng(0)
        yhat_c, yhat_t = rng.normal(size=1000), rng.normal(size=4000)
        y_c, y_t = yhat_c + rng.normal(scale=0.7, size=1000), yhat_t + rng.normal(scale=0.7, size=4000)
        cp = SplitConformal(0.1).calibrate(y_c, yhat_c)
        cov = SplitConformal.coverage(y_t, *cp.interval(yhat_t))
        self.assertTrue(0.88 <= cov <= 0.92, cov)

        class Passthrough(RegressorMixin, BaseEstimator):       # X column 0 is the prediction
            def fit(self, X, y=None):
                self.fitted_ = True
                return self

            def predict(self, X):
                return np.asarray(X)[:, 0]

        m = SplitConformalRegressor(Passthrough().fit(None), confidence_level=0.9, prefit=True)
        m.conformalize(yhat_c[:, None], y_c)
        _, iv = m.predict_interval(yhat_t[:, None])
        mapie_hw = float(np.median(iv[:, 1, 0] - yhat_t))
        self.assertAlmostEqual(mapie_hw, cp.q, delta=0.02 * cp.q)

    # ---- Block H + I
    def test_nsga3_pareto_front_and_resimulation_gap(self):
        cp, _ = self._cal(0.1)
        s = self.scene
        ev = SurrogateEvaluator(self.model, self.sg, self.flows, self.lengths, np.ones(s.n_segments), s.group,
                                s.population, conformal=cp, inequality="theil")
        budget = 8
        res = run_nsga3(ev, budget=budget, pop_size=30, n_gen=15, seed=0)
        X, F = res["X"], res["F"]
        self.assertGreater(len(X), 1)
        self.assertEqual(F.shape[1], 3)
        self.assertTrue(np.all(X.sum(1) <= budget))
        self.assertTrue(is_nondominated(F).all())
        self.assertGreater(len(np.unique(X.sum(1))), 1)            # spans tree budgets
        zero = ev(np.zeros((1, s.n_slots), int))[0]
        self.assertLess(F[:, 0].min(), zero[0])                     # trees reduce dose
        greedy = ev(greedy_allocation(ev, budget)[None])[0]
        self.assertLessEqual(F[:, 0].min(), 1.15 * greedy[0])       # competitive with greedy

        pick = X[np.argsort(F[:, 0])[:4]]
        gap = surrogate_optimality_gap(ev, pick, generate_labels(s, pick))
        self.assertLess(np.mean(np.abs(gap["rel_gap"])), 0.15)
        self.assertGreaterEqual(gap["upper_covers"].mean(), 0.75)


if __name__ == "__main__":
    unittest.main()
