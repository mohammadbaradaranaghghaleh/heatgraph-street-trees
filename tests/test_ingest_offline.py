"""Offline tests (stdlib only): no network, no heavy dependencies."""
import io
import math
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heatgraph.config import load_patches  # noqa: E402
from heatgraph.geo import approx_wgs84_bbox, utm_epsg, working_crs  # noqa: E402
from heatgraph.ingest.buildings_3dbag import parse_features  # noqa: E402
from heatgraph.ingest.run import main  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_patch_registry(self):
        meta, patches = load_patches()
        train = [p for p in patches if p.kind == "patch"]
        self.assertGreaterEqual(len(train), 15)
        self.assertLessEqual(len(train), 20)
        self.assertGreaterEqual(len({p.city for p in train}), 5)
        self.assertTrue(all(p.source == "3dbag" for p in train))
        self.assertTrue(meta["heatwave_days"])


class TestGeo(unittest.TestCase):
    def test_bbox_size(self):
        w, s, e, n = approx_wgs84_bbox(4.9, 52.37, 450)
        dy = (n - s) * 111_195
        dx = (e - w) * 111_195 * math.cos(math.radians(52.37))
        self.assertAlmostEqual(dy, 900, delta=1)
        self.assertAlmostEqual(dx, 900, delta=1)

    def test_crs(self):
        self.assertEqual(utm_epsg(7.85, 47.99), "EPSG:32632")
        self.assertEqual(working_crs("3dbag", 4.9, 52.37), "EPSG:28992")


class TestParse3DBAG(unittest.TestCase):
    def test_parse_minimal_feature(self):
        page = {"metadata": {"transform": {"scale": [0.001, 0.001, 0.001], "translate": [121000, 487000, 0]}},
                "features": [{"type": "CityJSONFeature", "vertices": [[0, 0, 0], [10000, 0, 0], [10000, 10000, 0], [0, 10000, 0]],
                              "CityObjects": {"B1": {"type": "Building",
                                                     "attributes": {"b3_h_maaiveld": 1.0, "b3_h_dak_70p": 13.0},
                                                     "geometry": [{"lod": "0", "type": "MultiSurface",
                                                                   "boundaries": [[[0, 1, 2, 3]]]}]}}}]}
        rows = parse_features(page)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["height_m"], 12.0)
        self.assertEqual(rows[0]["rings"][0][1], (121010.0, 487000.0))


class TestDryRun(unittest.TestCase):
    def test_dry_run_touches_nothing(self):
        raw = Path(__file__).resolve().parents[1] / "data" / "raw"

        def snapshot():
            return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in raw.rglob("*")) if raw.exists() else []

        before = snapshot()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([])
        self.assertEqual(rc, 0)
        self.assertIn("DRY-RUN", buf.getvalue())
        self.assertIn("network", buf.getvalue())          # Overture network is the default layer
        self.assertEqual(snapshot(), before)                # nothing created or modified


if __name__ == "__main__":
    unittest.main()
