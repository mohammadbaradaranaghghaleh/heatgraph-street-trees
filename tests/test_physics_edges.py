"""v2 physics-informed slot->segment edge features and early stopping (toy scene, CPU)."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE = all(importlib.util.find_spec(m) for m in ("numpy", "torch", "torch_geometric", "pvlib", "pythermalcomfort"))

if HAVE:
    import numpy as np

    from heatgraph.surrogate.graphs import EDGE_FEATURES, SceneGraph
    from heatgraph.synthetic import generate_labels, random_plantings, toy_scene

SHADE, OVER = (EDGE_FEATURES.index("crown_shade_frac"), EDGE_FEATURES.index("overhead_frac")) if HAVE else (0, 0)


@unittest.skipUnless(HAVE, "ML stack not installed")
class TestPhysicsEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = toy_scene(seed=0)
        cls.X = random_plantings(cls.s, 25, p=0.35, seed=2)
        cls.Y = generate_labels(cls.s, cls.X)
        cls.sg = SceneGraph(cls.s, baseline_utci=cls.Y[0])

    def test_ranges(self):
        a = self.sg.link_attr
        self.assertTrue(np.all((a[..., SHADE] >= 0) & (a[..., SHADE] <= 1)))
        self.assertTrue(np.all((a[..., OVER] >= 0) & (a[..., OVER] <= 1)))
        self.assertGreater(a[..., SHADE].max(), 0)                       # some crown shades some segment

    def test_zero_without_sun_and_when_disabled(self):
        low = self.sg.elev <= 3.0
        if low.any():
            self.assertTrue(np.all(self.sg.link_attr[low][..., SHADE] == 0))
        off = SceneGraph(self.s, baseline_utci=self.Y[0], physics_edges=False)
        self.assertTrue(np.all(off.link_attr[..., [SHADE, OVER]] == 0))
        np.testing.assert_allclose(off.link_attr[..., :4], self.sg.link_attr[..., :4])   # v1 columns unchanged

    def test_feature_tracks_engine_tree_effect(self):
        sg, E = self.sg, self.s.n_segments
        xs, ys = [], []
        for i in range(1, len(self.X)):
            keep = self.X[i][sg.link_slot] > 0
            for h in range(sg.hours):
                xs.append(np.bincount(sg.link_seg[keep], weights=sg.link_attr[h][keep, SHADE], minlength=E))
                ys.append(self.Y[i, h] - self.Y[0, h])
        r = np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]
        self.assertLess(r, -0.7, r)                                       # more crown shadow -> cooler

    def test_early_stopping(self):
        from heatgraph.surrogate.model import HeatGraphGNN
        from heatgraph.surrogate.train import fit

        tr, va = self.sg.dataset(self.X[:15], self.Y[:15]), self.sg.dataset(self.X[15:], self.Y[15:])
        h = fit(HeatGraphGNN(hidden=16), tr, va, epochs=200, lr=1e-2, patience=3)
        self.assertIn("best_epoch", h)
        self.assertLess(len(h["train_mae"]), 200)
        self.assertEqual(h["stopped_epoch"] - h["best_epoch"], 3)


if __name__ == "__main__":
    unittest.main()
