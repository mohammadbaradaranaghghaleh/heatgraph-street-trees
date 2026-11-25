"""Offline tests for real-patch helpers: UMEP met file, local-day selection, Tmrt comparison."""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "shapely", "pyproj"))

if HAVE_DEPS:
    import numpy as np
    import pandas as pd

    from heatgraph.realpatch import UMEP_HEADER, compare_tmrt, local_day_rows, umep_metfile


def toy_weather():
    t = pd.date_range("2019-07-24 20:00", "2019-07-25 23:00", freq="h", tz="UTC")
    n = len(t)
    return pd.DataFrame({"temperature_2m": np.linspace(20, 30, n), "relative_humidity_2m": 50.0,
                         "wind_speed_10m": 2.0, "surface_pressure": 1013.0, "shortwave_radiation": 500.0,
                         "diffuse_radiation": 100.0, "direct_normal_irradiance": 600.0}, index=t)


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed (use HeatGraph/.venv)")
class TestRealPatchHelpers(unittest.TestCase):
    def test_local_day_is_cest(self):
        rows = local_day_rows(toy_weather(), "2019-07-25")
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows.index[0], pd.Timestamp("2019-07-24 22:00", tz="UTC"))   # 00:00 CEST

    def test_umep_metfile_columns(self):
        rows = local_day_rows(toy_weather(), "2019-07-25")
        lines = umep_metfile(rows).strip().splitlines()
        self.assertEqual(lines[0], UMEP_HEADER)
        self.assertEqual(len(lines), 25)
        first = lines[1].split()
        self.assertEqual(len(first), 24)
        self.assertEqual(first[:4], ["2019", "206", "0", "0"])        # local midnight, DOY 206
        v = [float(x) for x in first]
        self.assertEqual(v[9], 2.0)          # U
        self.assertEqual(v[10], 50.0)        # RH
        self.assertAlmostEqual(v[12], 101.3)  # kPa
        self.assertEqual((v[14], v[21], v[22]), (500.0, 100.0, 600.0))
        self.assertEqual(v[4], -999.0)

    def test_compare_tmrt(self):
        ref = np.zeros((2, 4, 4)) + 40
        ours = ref + np.array([1.0, -3.0])[:, None, None]
        mask = np.ones((4, 4), bool); mask[0, 0] = False
        sunlit = np.zeros((2, 4, 4)); sunlit[1] = 1
        r = compare_tmrt(ours, ref, mask, sunlit)
        self.assertEqual(r["n"], 30)
        self.assertAlmostEqual(r["mae"], 2.0)
        self.assertAlmostEqual(r["bias"], -1.0)
        self.assertAlmostEqual(r["sunlit"]["bias"], -3.0)
        self.assertAlmostEqual(r["shaded"]["bias"], 1.0)


if __name__ == "__main__":
    unittest.main()
