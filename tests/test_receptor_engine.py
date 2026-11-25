"""The receptor/batched (GPU) engine must reproduce the full-grid numpy engine exactly
(up to float32 rounding) on the toy scene, for several plantings at once."""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "torch", "pvlib", "pythermalcomfort"))

if HAVE_DEPS:
    import numpy as np
    import torch

    from heatgraph.engine import svf as svf_mod
    from heatgraph.engine.receptor_engine import ReceptorEngine, segment_mean
    from heatgraph.engine.simulate import simulate
    from heatgraph.engine.surfaces import grids_from_arrays
    from heatgraph.synthetic import TOY_ENGINE_CFG, random_plantings, simulate_planting, toy_scene


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed (use HeatGraph/.venv)")
class TestReceptorEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = s = toy_scene(seed=0)
        cls.rc = np.vstack(s.receptors)
        cls.seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
        cls.X = random_plantings(s, 4, p=0.4, seed=5)

    def _engine(self, device):
        return ReceptorEngine(self.s.building_heights, self.rc, self.s.weather, self.s.lat, self.s.lon,
                              TOY_ENGINE_CFG, device=device)

    def _check(self, device):
        eng = self._engine(device)
        cd = np.stack([grids_from_arrays(self.s.building_heights, self.s.canopy_with(x)).cdsm for x in self.X])
        out = eng.utci(cd)
        for i, x in enumerate(self.X):
            ref = simulate_planting(self.s, x)
            sim = ref["sim"]
            r, c = self.rc[:, 0], self.rc[:, 1]
            np.testing.assert_allclose(out["svf"][i], sim["svf"][r, c], atol=2e-5)
            np.testing.assert_allclose(out["sunlit"][i], sim["sunlit"][:, r, c], atol=1e-6)
            np.testing.assert_allclose(out["tmrt"][i], sim["tmrt"][:, r, c], atol=2e-3)
            seg_utci = segment_mean(out["utci"][i], self.seg, self.s.n_segments)
            np.testing.assert_allclose(seg_utci, ref["utci"], atol=2e-3)

    def test_matches_full_grid_engine_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(), "CUDA not available")
    def test_matches_full_grid_engine_cuda(self):
        self._check("cuda")

    def test_segment_mean(self):
        v = np.array([[1.0, 3.0, 10.0]])
        np.testing.assert_allclose(segment_mean(v, np.array([0, 0, 1]), 2), [[2.0, 10.0]])


if __name__ == "__main__":
    unittest.main()
