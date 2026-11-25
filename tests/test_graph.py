"""Block D tests: flows, vulnerability, exposure dose and equity metrics on toy graphs."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "networkx"))

if HAVE_DEPS:
    import networkx as nx
    import numpy as np

    from heatgraph.graph import equity, exposure, flows, vulnerability


def path_graph():
    G = nx.Graph()
    G.add_edge("A", "B", length=100.0)
    G.add_edge("B", "C", length=100.0)
    return G


@unittest.skipUnless(HAVE_DEPS, "numpy/networkx not installed (use HeatGraph/.venv)")
class TestFlows(unittest.TestCase):
    def test_all_trips_follow_the_only_path(self):
        f = flows.gravity_flows(path_graph(), {"A": 100}, {"C": 1.0})
        self.assertAlmostEqual(f[("A", "B")], 100)
        self.assertAlmostEqual(f[("B", "C")], 100)

    def test_equal_destinations_split_evenly(self):
        G = path_graph()
        f = flows.gravity_flows(G, {"B": 80}, {"A": 1.0, "C": 1.0})
        self.assertAlmostEqual(f[("A", "B")], 40)
        self.assertAlmostEqual(f[("B", "C")], 40)

    def test_distance_decay_favours_near_destination(self):
        G = nx.Graph()
        G.add_edge("O", "N", length=100.0); G.add_edge("O", "F", length=500.0)
        f = flows.gravity_flows(G, {"O": 100}, {"N": 1.0, "F": 1.0}, beta=0.002)
        near, far = f[("N", "O")], f[("F", "O")]
        self.assertAlmostEqual(near + far, 100)
        self.assertAlmostEqual(near / far, math.exp(0.002 * 400), places=6)

    def test_detour_paths_share_trips(self):
        G = nx.cycle_graph(["A", "B", "C", "D"])
        nx.set_edge_attributes(G, 100.0, "length")
        f = flows.gravity_flows(G, {"A": 10}, {"C": 1.0}, detour_ratio=1.15)
        for e in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]:
            self.assertAlmostEqual(f[e], 5.0)

    def test_detour_limit_excludes_long_path(self):
        G = nx.Graph()
        G.add_edge("A", "C", length=100.0)
        G.add_edge("A", "B", length=100.0); G.add_edge("B", "C", length=100.0)   # 2x longer
        f = flows.gravity_flows(G, {"A": 10}, {"C": 1.0}, detour_ratio=1.15)
        self.assertAlmostEqual(f[("A", "C")], 10)
        self.assertAlmostEqual(f[("A", "B")], 0)

    def test_radius_cutoff(self):
        f = flows.gravity_flows(path_graph(), {"A": 100}, {"C": 1.0}, radius_m=150)
        self.assertEqual(sum(f.values()), 0)

    def test_multigraph_and_betweenness(self):
        G = nx.MultiGraph()
        G.add_edge("A", "B", length=100.0); G.add_edge("A", "B", length=300.0); G.add_edge("B", "C", length=50.0)
        f = flows.gravity_flows(G, {"A": 1}, {"C": 1.0})
        self.assertAlmostEqual(f[("A", "B")], 1)
        eb = flows.betweenness_flows(path_graph())
        self.assertEqual(set(eb), {("A", "B"), ("B", "C")})


@unittest.skipUnless(HAVE_DEPS, "numpy/networkx not installed")
class TestVulnerabilityAndDose(unittest.TestCase):
    def test_vulnerability(self):
        v = vulnerability.vulnerability_index([100, 100, 0], [10, 0, 0], [20, 50, 0], w_child=1, w_elderly=2)
        np.testing.assert_allclose(v, [1 + 0.1 + 0.4, 1 + 1.0, 1.0])
        with self.assertRaises(ValueError):
            vulnerability.vulnerability_index([10], [8], [5])

    def test_dose_hand_calculation(self):
        utci = np.array([[30.0, 20.0], [36.0, 40.0]])     # 2 hours x 2 edges
        flow = np.array([[100.0, 50.0], [10.0, 50.0]])
        length = np.array([130.0, 260.0])                  # 100 s and 200 s at 1.3 m/s
        d = exposure.edge_heat_dose(utci, flow, length, vulnerability=np.array([1.0, 2.0]))
        e0 = (100 * 4 + 10 * 10) * 100 / 3600
        e1 = (50 * 0 + 50 * 14) * 200 / 3600 * 2
        np.testing.assert_allclose(d, [e0, e1])

    def test_dose_rejects_nan(self):
        with self.assertRaises(ValueError):
            exposure.edge_heat_dose(np.array([[np.nan]]), np.array([[1.0]]), np.array([10.0]))

    def test_receptors_and_aggregation(self):
        rc = exposure.sample_receptors_along_line(np.array([[0.5, 2.5], [20.5, 2.5]]), 5.0, lambda x, y: (x, y))
        self.assertEqual(len(rc), 4)
        self.assertTrue(np.all(rc[:, 0] == 2))
        grid = np.arange(2 * 5 * 25, dtype=float).reshape(2, 5, 25)
        out = exposure.edge_utci_from_receptors(grid, [rc])
        self.assertEqual(out.shape, (2, 1))
        self.assertAlmostEqual(out[0, 0], grid[0, rc[:, 0], rc[:, 1]].mean())


@unittest.skipUnless(HAVE_DEPS, "numpy/networkx not installed")
class TestEquity(unittest.TestCase):
    def test_gini_bounds(self):
        self.assertAlmostEqual(equity.gini([5, 5, 5, 5]), 0.0)
        self.assertAlmostEqual(equity.gini([0, 0, 0, 8]), 0.75)
        self.assertAlmostEqual(equity.gini([1, 2, 3, 4]), 0.25)   # known value for 1..4

    def test_weights_equal_replication(self):
        self.assertAlmostEqual(equity.gini([1, 3], [2, 1]), equity.gini([1, 1, 3]))
        self.assertAlmostEqual(equity.theil([1, 3], [2, 1]), equity.theil([1, 1, 3]))

    def test_theil_properties(self):
        self.assertAlmostEqual(equity.theil([2, 2, 2]), 0.0)
        self.assertAlmostEqual(equity.theil([0, 0, 0, 7]), math.log(4))
        x = np.array([1.0, 4.0, 2.0, 9.0])
        self.assertAlmostEqual(equity.theil(x), equity.theil(10 * x))  # scale invariant

    def test_theil_decomposition_adds_up(self):
        x = np.array([1.0, 2.0, 6.0, 8.0, 3.0])
        w = np.array([10, 20, 5, 5, 10])
        d = equity.theil_decomposition(x, ["a", "a", "b", "b", "c"], w)
        self.assertAlmostEqual(d["total"], d["between"] + d["within"], places=12)
        self.assertGreater(d["between"], 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            equity.gini([-1, 2])


if __name__ == "__main__":
    unittest.main()
