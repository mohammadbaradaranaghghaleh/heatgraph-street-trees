"""Stage 5 components on the toy scene: fast batch builder, linear surrogate, additive evaluator,
fast single-path flows."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE = all(importlib.util.find_spec(m) for m in ("numpy", "torch", "torch_geometric", "pvlib", "pythermalcomfort",
                                                  "networkx"))

if HAVE:
    import networkx as nx
    import numpy as np
    import torch
    from torch_geometric.data import Batch

    from heatgraph.graph.flows import gravity_flows
    from heatgraph.optimize.additive import AdditiveEvaluator
    from heatgraph.surrogate.graphs import SceneGraph
    from heatgraph.surrogate.linear import LinearPhysicsSurrogate
    from heatgraph.surrogate.model import HeatGraphGNN
    from heatgraph.synthetic import generate_labels, random_plantings, toy_scene


@unittest.skipUnless(HAVE, "ML stack not installed")
class TestStage5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.s = toy_scene(seed=0)
        cls.X = random_plantings(cls.s, 12, p=0.4, seed=3)
        cls.Y = generate_labels(cls.s, cls.X)
        cls.sg = SceneGraph(cls.s, baseline_utci=cls.Y[0])
        cls.lin = LinearPhysicsSurrogate.fit([(cls.sg, cls.X, cls.Y)])

    def test_build_batch_matches_from_data_list(self):
        X = self.X[1:5]
        b1 = Batch.from_data_list([self.sg.build(x, h) for x in X for h in range(self.sg.hours)])
        b2 = self.sg.build_batch(X)
        m = HeatGraphGNN(hidden=16).eval()
        with torch.no_grad():
            np.testing.assert_allclose(m.predict(b1).numpy(), m.predict(b2).numpy(), atol=1e-5)
        np.testing.assert_allclose(self.lin.predict(b1).numpy(), self.lin.predict(b2).numpy(), atol=1e-6)

    def test_additive_evaluator_equals_graph_linear(self):
        E = self.s.n_segments
        ev = AdditiveEvaluator(self.sg, self.lin.coef.numpy(), np.ones((self.sg.hours, E)), np.full(E, 30.0),
                               np.ones(E), self.s.group, self.s.population)
        fast = ev.predict_utci(self.X, "point")
        graph = self.lin.predict(self.sg.build_batch(self.X)).numpy().reshape(len(self.X), self.sg.hours, E)
        np.testing.assert_allclose(fast, graph, atol=1e-4)
        np.testing.assert_allclose(fast[0], self.Y[0], atol=1e-5)          # no trees -> engine baseline

    def test_linear_surrogate_learns_tree_effect(self):
        P = self.lin.predict(self.sg.build_batch(self.X)).numpy().reshape(self.Y.shape)
        dt, dp = (self.Y - self.Y[0]).ravel(), (P - self.Y[0]).ravel()
        self.assertGreater(np.corrcoef(dt, dp)[0, 1], 0.9)

    def test_fast_flows_equal_general_on_tree_network(self):
        G = nx.balanced_tree(2, 3)                                         # unique shortest paths
        nx.set_edge_attributes(G, 50.0, "length")
        o = {n: 1.0 for n in G}
        d = {n: float(n % 3 + 1) for n in G}
        f1 = gravity_flows(G, o, d, radius_m=1000, max_paths=1)
        f2 = gravity_flows(G, o, d, radius_m=1000, max_paths=10, detour_ratio=1.0)
        for k in f1:
            self.assertAlmostEqual(f1[k], f2[k], places=9)


if __name__ == "__main__":
    unittest.main()
