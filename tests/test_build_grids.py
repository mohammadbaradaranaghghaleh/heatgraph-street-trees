"""build_grids / rasterize_polygons on synthetic footprints (numpy + shapely only)."""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "shapely", "pvlib"))

if HAVE_DEPS:
    import numpy as np
    from shapely.geometry import Polygon, box

    from heatgraph.engine import shadow
    from heatgraph.engine.surfaces import GridSpec, build_grids, impute_missing_heights, rasterize_polygons

BOUNDS = (90.0, 190.0, 130.0, 230.0)   # 40 m x 40 m, maxy = 230


@unittest.skipUnless(HAVE_DEPS, "numpy/shapely not installed (use HeatGraph/.venv)")
class TestRasterize(unittest.TestCase):
    def test_axis_aligned_square_exact(self):
        spec = GridSpec.from_bounds(BOUNDS, 1.0)
        a = rasterize_polygons([box(100, 200, 110, 210)], [12.0], spec)
        self.assertEqual(int((a > 0).sum()), 100)
        rows, cols = np.nonzero(a)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (20, 29, 10, 19))
        self.assertTrue(np.all(a[a > 0] == 12.0))

    def test_resolution(self):
        a = rasterize_polygons([box(100, 200, 110, 210)], [5.0], GridSpec.from_bounds(BOUNDS, 2.0))
        self.assertEqual(int((a > 0).sum()), 25)

    def test_hole_is_respected(self):
        ring = Polygon(box(95, 195, 115, 215).exterior.coords, [box(100, 200, 110, 210).exterior.coords])
        a = rasterize_polygons([ring], [9.0], GridSpec.from_bounds(BOUNDS, 1.0))
        self.assertEqual(int((a > 0).sum()), 400 - 100)

    def test_triangle_area_close(self):
        tri = Polygon([(95, 195), (115, 195), (95, 215)])       # area 200 m2
        a = rasterize_polygons([tri], [3.0], GridSpec.from_bounds(BOUNDS, 1.0))
        self.assertAlmostEqual(int((a > 0).sum()), 200, delta=20)

    def test_overlap_takes_max_and_clipping(self):
        a = rasterize_polygons([box(100, 200, 110, 210), box(105, 205, 140, 240)], [10.0, 25.0],
                               GridSpec.from_bounds(BOUNDS, 1.0))
        spec = GridSpec.from_bounds(BOUNDS, 1.0)
        c, r = spec.xy_to_colrow(107.5, 207.5)
        self.assertEqual(a[int(r), int(c)], 25.0)                # overlap -> taller
        c, r = spec.xy_to_colrow(101.5, 201.5)
        self.assertEqual(a[int(r), int(c)], 10.0)
        self.assertEqual(a.shape, (40, 40))                      # out-of-bounds part clipped

    def test_centres_roundtrip(self):
        spec = GridSpec.from_bounds(BOUNDS, 1.0)
        X, Y = spec.centres()
        c, r = spec.xy_to_colrow(X, Y)
        self.assertTrue(np.allclose(np.floor(c), np.arange(40)[None, :]))
        self.assertTrue(np.allclose(np.floor(r), np.arange(40)[:, None]))
        self.assertEqual(spec.bounds, BOUNDS)


@unittest.skipUnless(HAVE_DEPS, "numpy/shapely not installed")
class TestBuildGrids(unittest.TestCase):
    def test_imputation(self):
        geoms = [box(0, 0, 5, 5), box(10, 0, 15, 5), box(20, 0, 25, 5), box(500, 500, 505, 505), box(12, 10, 17, 15)]
        h, imp = impute_missing_heights(geoms, [10.0, 12.0, 14.0, 40.0, None])
        self.assertEqual(h[4], 12.0)                              # median of near neighbours, not 40
        self.assertEqual(imp.tolist(), [False, False, False, False, True])
        with self.assertRaises(ValueError):
            impute_missing_heights(geoms[:2], [None, None])

    def test_build_grids_end_to_end(self):
        bld = [(box(100, 200, 110, 210), 15.0), (box(115, 200, 120, 205), None)]
        canopy = np.zeros((40, 40), np.float32)
        canopy[22:25, 12:15] = 9.0                                # partly under the building -> masked
        canopy[5:8, 30:33] = 7.0
        g = build_grids(bld, BOUNDS, 1.0, canopy=canopy, crs="EPSG:28992")
        self.assertEqual(g.dsm.shape, (40, 40))
        self.assertEqual(g.n_imputed, 1)
        self.assertEqual(float(g.dsm[22, 12]), 15.0)
        self.assertEqual(float(g.cdsm[22, 12]), 0.0)             # no canopy inside buildings
        self.assertEqual(float(g.cdsm[6, 31]), 7.0)
        self.assertGreater(float(g.wall_height.max()), 0)
        self.assertEqual(g.transform.res, 1.0)
        # the rasterised building casts a northward shadow with the sun in the south
        s, _, _ = shadow.cast(g.dsm, g.cdsm, g.tdsm, 180.0, 45.0)
        self.assertEqual(s[15, 14], 0.0)                         # 5 m north of the building
        self.assertEqual(s[35, 14], 1.0)                         # south side sunlit

    def test_canopy_shape_mismatch(self):
        with self.assertRaises(ValueError):
            build_grids([(box(100, 200, 110, 210), 5.0)], BOUNDS, 1.0, canopy=np.zeros((10, 10)))


if __name__ == "__main__":
    unittest.main()
