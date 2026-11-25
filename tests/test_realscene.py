"""Real-scene invariants on an ingested patch (skipped when data/raw is not present)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PATCH = "utr_binnenstad"
HAVE = all(importlib.util.find_spec(m) for m in ("numpy", "geopandas", "rasterio", "pvlib")) and \
    (ROOT / "data" / "raw" / PATCH / "network.gpkg").exists()


@unittest.skipUnless(HAVE, "ingested patch data or geo stack not available")
class TestRealScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heatgraph.config import load_toml
        from heatgraph.realpatch import load_patch
        from heatgraph.realscene import build_scene

        cls.p = load_patch(PATCH, load_toml("engine.toml"))
        cls.s = build_scene(cls.p)

    def test_receptors_and_slots_are_valid(self):
        import numpy as np

        s = self.s
        rc = np.vstack(s.receptors)
        self.assertTrue(np.all(s.building_heights[rc[:, 0], rc[:, 1]] <= 0))          # receptors on open ground
        self.assertTrue(np.all(self.p.core_mask[s.slots[:, 0], s.slots[:, 1]]))       # slots in the core
        self.assertFalse(s.water_mask[s.slots[:, 0], s.slots[:, 1]].any())            # not in water
        d = np.hypot(*(s.slots[:, None, :] - s.slots[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        self.assertGreaterEqual(d.min(), 8.0)                                         # min slot spacing
        self.assertEqual(len(s.weather), 36)                                          # 12 h x 3 days

    def test_canopy_with_respects_buildings(self):
        import numpy as np

        s = self.s
        full = s.canopy_with(np.ones(s.n_slots, int))
        self.assertTrue(np.all(full[s.building_heights > 0] == 0))
        self.assertTrue(np.all(full[s.slots[:, 0], s.slots[:, 1]] >= s.crown_height))
        self.assertTrue(np.all(full >= np.where(s.building_heights > 0, 0, s.base_canopy)))

    def test_save_load_roundtrip(self):
        import numpy as np

        from heatgraph.realscene import load_scene, save_scene

        with tempfile.TemporaryDirectory() as d:
            save_scene(self.s, d)
            t = load_scene(d)
        self.assertEqual(t.n_segments, self.s.n_segments)
        self.assertEqual(t.n_slots, self.s.n_slots)
        np.testing.assert_array_equal(np.vstack(t.receptors), np.vstack(self.s.receptors))
        np.testing.assert_array_equal(t.canopy_with(np.ones(t.n_slots, int)), self.s.canopy_with(np.ones(t.n_slots, int)))
        self.assertTrue((t.weather.index == self.s.weather.index).all())


if __name__ == "__main__":
    unittest.main()
