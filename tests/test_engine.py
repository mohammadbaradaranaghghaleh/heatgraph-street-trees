"""Block C engine tests on synthetic scenes with analytic or physically obvious answers.

Grid convention: row 0 = north, rows increase southwards, cols increase eastwards, 1 m pixels.
Needs numpy/pandas/pvlib/pythermalcomfort (HeatGraph/.venv); skipped otherwise.
"""
import importlib.util
import math
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pvlib", "pythermalcomfort"))

if HAVE_DEPS:
    import numpy as np
    import pandas as pd

    from heatgraph.engine import comfort, radiation, shadow, sun, svf
    from heatgraph.engine.simulate import simulate
    from heatgraph.engine.surfaces import grids_from_arrays


def cube_scene(n=101, h=10.0, r0=20, r1=25, c0=20, c1=25):
    b = np.zeros((n, n), np.float32)
    b[r0:r1, c0:c1] = h
    return b


def tree_scene(n=61, top=10.0, radius=3, centre=30):
    yy, xx = np.mgrid[0:n, 0:n]
    crown = (yy - centre) ** 2 + (xx - centre) ** 2 <= radius ** 2
    return np.where(crown, top, 0.0).astype(np.float32)


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed (use HeatGraph/.venv)")
class TestSun(unittest.TestCase):
    def test_solstice_noon_elevation_amsterdam(self):
        times = pd.date_range("2022-06-21 10:00", "2022-06-21 13:00", freq="1min", tz="UTC")
        pos = sun.positions(times, 52.37, 4.90)
        peak = pos["elevation"].idxmax()
        expected = 90 - 52.37 + 23.44            # noon elevation at the June solstice
        self.assertAlmostEqual(pos["elevation"].max(), expected, delta=0.3)
        self.assertAlmostEqual(pos.loc[peak, "azimuth"], 180.0, delta=1.0)

    def test_night_and_sun_vector(self):
        pos = sun.positions(pd.DatetimeIndex(["2022-06-21 23:00"], tz="UTC"), 52.37, 4.90)
        self.assertLess(pos["elevation"].iloc[0], 0)
        dr, dc = sun.sun_vector_grid(180.0)    # sun in the south -> step southwards (+row)
        self.assertAlmostEqual(dr, 1.0)
        self.assertAlmostEqual(dc, 0.0, places=12)
        dr, dc = sun.sun_vector_grid(90.0)     # sun in the east -> +col
        self.assertAlmostEqual(dc, 1.0)

    def test_erbs_split(self):
        t = pd.DatetimeIndex(["2022-06-21 12:00", "2022-06-21 23:00"], tz="UTC")
        dni, dhi, ghi = sun.split_radiation([850.0, 0.0], [30.0, 110.0], t)
        self.assertGreater(dni[0], 400)
        self.assertLessEqual(dhi[0], ghi[0])
        self.assertEqual(dni[1], 0.0)


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed")
class TestShadow(unittest.TestCase):
    def test_flat_ground_fully_sunlit(self):
        z = np.zeros((40, 40), np.float32)
        s, b, v = shadow.cast(z, None, None, 135.0, 20.0)
        self.assertTrue(np.all(s == 1.0) and not b.any() and not v.any())

    def test_cube_shadow_length_sun_south_45deg(self):
        dsm = cube_scene()                       # 10 m cube occupying rows 20-24
        s, b, _ = shadow.cast(dsm, None, None, 180.0, 45.0)
        col = 22
        for r in range(11, 20):                  # distance to cube < 10 m -> shaded
            self.assertEqual(s[r, col], 0.0, f"row {r} should be shaded")
        self.assertEqual(s[9, col], 1.0)         # 11 m away: ray clears the 10 m roof
        self.assertEqual(s[30, col], 1.0)        # south side faces the sun
        self.assertEqual(s[15, 30], 1.0)         # beside the shadow strip

    def test_shadow_length_scales_with_elevation(self):
        dsm = cube_scene()
        s, _, _ = shadow.cast(dsm, None, None, 180.0, 30.0)   # length 10/tan30 = 17.3 m
        self.assertEqual(s[3, 22], 0.0)
        self.assertEqual(s[1, 22], 1.0)

    def test_sun_from_east_casts_west(self):
        dsm = cube_scene()
        s, _, _ = shadow.cast(dsm, None, None, 90.0, 45.0)
        self.assertEqual(s[22, 15], 0.0)         # west of the cube
        self.assertEqual(s[22, 30], 1.0)         # east of the cube

    def test_tree_shadow_is_transmissive_and_bounded(self):
        cdsm = tree_scene()
        g = grids_from_arrays(np.zeros_like(cdsm), cdsm, trunk_zone_fraction=0.25)
        s, b, v = shadow.cast(g.dsm, g.cdsm, g.tdsm, 180.0, 45.0, tree_transmissivity=0.03)
        self.assertFalse(b.any())
        self.assertAlmostEqual(float(s[24, 30]), 0.03, places=6)   # ray crosses crown at 3-9 m height
        self.assertEqual(s[12, 30], 1.0)                           # ray passes over the 10 m crown
        self.assertEqual(s[40, 30], 1.0)                           # south of the tree

    def test_building_shadow_overrides_tree(self):
        dsm = cube_scene()
        cdsm = np.zeros_like(dsm); cdsm[14:17, 21:24] = 6.0
        g = grids_from_arrays(dsm, cdsm)
        s, b, v = shadow.cast(g.dsm, g.cdsm, g.tdsm, 180.0, 45.0)
        self.assertEqual(s[12, 22], 0.0)
        self.assertFalse((b & v).any())

    def test_night(self):
        s, b, _ = shadow.cast(cube_scene(), None, None, 0.0, -5.0)
        self.assertTrue(np.all(s == 0) and b.all())


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed")
class TestSVF(unittest.TestCase):
    def test_open_ground_is_one(self):
        out = svf.compute(np.zeros((30, 30), np.float32), n_azimuths=16, max_distance_m=30)
        self.assertTrue(np.allclose(out["svf"], 1.0))

    def test_canyon_matches_analytic(self):
        H, half = 10.0, 11                      # wall pixel centres 11 m from the receptor
        n_rows, width = 401, 31
        dsm = np.zeros((n_rows, width), np.float32)
        dsm[:, :5] = H; dsm[:, 26:] = H          # floor cols 5..25, receptor col 15
        out = svf.compute(dsm, n_azimuths=72, max_distance_m=200)
        expected = svf.canyon_svf_analytic(H, 2 * half)
        self.assertAlmostEqual(float(out["svf_b"][200, 15]), expected, delta=0.02)

    def test_canopy_reduces_svf(self):
        cdsm = tree_scene(n=41, top=8.0, radius=4, centre=20)
        out = svf.compute(np.zeros_like(cdsm), cdsm, n_azimuths=36, max_distance_m=40,
                          tree_transmissivity=0.03)
        under, open_ = (20, 20), (2, 2)
        self.assertLess(out["svf"][under], 0.5)
        self.assertLess(out["svf_bv"][under], out["svf_b"][under])
        self.assertTrue(out["svf_bv"][under] <= out["svf"][under] <= out["svf_b"][under])
        self.assertGreater(out["svf"][open_], out["svf"][under])


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed")
class TestTmrtUtci(unittest.TestCase):
    noon = dict(ta_c=30.0, rh_pct=40.0, dni=800.0, dhi=120.0, elevation_deg=60.0)

    def test_sunlit_open_square_plausible(self):
        t = radiation.tmrt(1.0, 1.0, **self.noon)
        self.assertTrue(50.0 < float(t) < 75.0, t)

    def test_shade_and_canopy_cool(self):
        sunlit = float(radiation.tmrt(1.0, 0.8, **self.noon))
        bshade = float(radiation.tmrt(0.0, 0.5, **self.noon))
        canopy = float(radiation.tmrt(0.03, 0.2, **self.noon))
        self.assertGreater(sunlit - bshade, 15.0)
        self.assertGreater(sunlit - canopy, 15.0)

    def test_night_below_air_temperature_under_open_sky(self):
        t = float(radiation.tmrt(0.0, 1.0, 20.0, 60.0, 0.0, 0.0, -10.0))
        self.assertLess(t, 20.0)

    def test_tmrt_monotonic_in_beam(self):
        a = [float(radiation.tmrt(1.0, 1.0, 30, 40, d, 120, 60)) for d in (200, 500, 800)]
        self.assertTrue(a[0] < a[1] < a[2])

    def test_utci_sensitivity(self):
        base = comfort.utci_grid(30.0, np.array([30.0, 45.0, 60.0]), 2.0, 40.0)
        self.assertTrue(np.all(np.diff(base) > 0))                        # increases with Tmrt
        warmer = comfort.utci_grid(np.array([25.0, 30.0, 35.0]), 50.0, 2.0, 40.0)
        self.assertTrue(np.all(np.diff(warmer) > 0))                      # increases with Ta
        windy = comfort.utci_grid(30.0, 60.0, np.array([0.5, 4.0]), 40.0)
        self.assertGreater(windy[0], windy[1])                            # wind cools when hot
        calm = comfort.utci_grid(30.0, 60.0, 0.1, 40.0)
        self.assertTrue(np.isfinite(calm))                                # clipped, not NaN
        self.assertEqual(comfort.utci_grid(np.zeros((3, 4)) + 30, 50, 2, 40).shape, (3, 4))

    def test_wind_profile_helper(self):
        self.assertAlmostEqual(comfort.wind_at_height(5.0, 10.0, 10.0), 5.0)
        self.assertLess(comfort.wind_at_height(5.0), 5.0)


@unittest.skipUnless(HAVE_DEPS, "scientific stack not installed")
class TestSimulateIntegration(unittest.TestCase):
    def test_small_scene_heatwave_hours(self):
        with open(ROOT / "configs" / "engine.toml", "rb") as f:
            cfg = tomllib.load(f)
        cfg["svf"]["n_azimuths"], cfg["svf"]["max_distance_m"], cfg["grid"]["buffer_m"] = 16, 60, 60
        b = np.zeros((61, 61), np.float32); b[25:35, 25:35] = 15.0
        c = tree_scene(n=61, top=8.0, radius=3, centre=48)
        g = grids_from_arrays(b, c)
        times = pd.DatetimeIndex(["2019-07-25 06:00", "2019-07-25 11:40", "2019-07-25 22:00"], tz="UTC")
        wx = pd.DataFrame({"temperature_2m": [24.0, 36.0, 28.0], "relative_humidity_2m": [60.0, 30.0, 50.0],
                           "wind_speed_10m": [2.0, 3.0, 1.5], "shortwave_radiation": [350.0, 850.0, 0.0]},
                          index=times)
        out = simulate(g, wx, 52.37, 4.90, cfg)
        self.assertEqual(out["utci"].shape, (3, 61, 61))
        noon = out["utci"][1]
        north_shade, open_south = noon[20, 30], noon[45, 30]   # north of block is shaded at noon
        self.assertLess(north_shade, open_south - 3.0)
        self.assertTrue(np.isfinite(out["utci"][:, :24, :]).all())
        self.assertTrue(np.all(out["sunlit"][2] == 0))          # night


if __name__ == "__main__":
    unittest.main()
