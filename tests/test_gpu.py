"""GPU regression test (skipped without CUDA): training on CUDA must not move the caller's data.

Guards the bug found in Stage 3: HeteroData.to() is in-place, so a warm-up call in fit()
moved train_data[0] to the GPU and later CPU-side collation failed.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
CUDA = False
if HAVE_TORCH:
    import torch
    CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "CUDA not available")
class TestGPUTraining(unittest.TestCase):
    def test_fit_on_cuda_keeps_data_on_cpu(self):
        from heatgraph.surrogate.graphs import SceneGraph
        from heatgraph.surrogate.model import HeatGraphGNN
        from heatgraph.surrogate.train import evaluate, fit
        from heatgraph.synthetic import generate_labels, random_plantings, toy_scene

        s = toy_scene()
        X = random_plantings(s, 12, 0.35, 3)
        Y = generate_labels(s, X)
        sg = SceneGraph(s, baseline_utci=Y[0])
        tr, va = sg.dataset(X[:9], Y[:9]), sg.dataset(X[9:], Y[9:])
        m = HeatGraphGNN(hidden=16)
        fit(m, tr, va, epochs=3, device="cuda")
        self.assertEqual({p.device.type for p in m.parameters()}, {"cuda"})
        self.assertEqual({d[k].x.device.type for d in tr for k in ("segment", "slot")}, {"cpu"})
        self.assertTrue(abs(evaluate(m, va, device="cuda")["mae"]) < 50)


if __name__ == "__main__":
    unittest.main()
