"""Headless smoke test of app.py for every test patch and every budget (skips without dashboard data/streamlit)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard_data"
HAVE = importlib.util.find_spec("streamlit") is not None and (DASH / "rot_centrum" / "results.json").exists()
PATCHES = ["rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt"]


@unittest.skipUnless(HAVE, "streamlit or saved Stage 5 results not available")
class TestDashboard(unittest.TestCase):
    def run_app(self, patch, budget_index=None, hour_index=0, right="random"):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.run()
        at.sidebar.selectbox[0].set_value(patch).run()
        if budget_index is not None:
            opts = at.sidebar.select_slider[0].options
            at.sidebar.select_slider[0].set_value(int(str(opts[budget_index]).split()[0])).run()   # label "12 trees (5% ...)"
        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[hour_index]).run()
        at.sidebar.selectbox[3].set_value(right).run()
        return at

    def test_all_patches_render_without_exceptions(self):
        for p in PATCHES:
            with self.subTest(patch=p):
                at = self.run_app(p)
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertGreaterEqual(len(at.metric), 6)

    def test_budgets_hours_and_methods(self):
        for bi, hi, right in ((0, 5, "greedy"), (4, 0, "nsga_linear")):
            with self.subTest(budget=bi, hour=hi):
                at = self.run_app("rot_centrum", budget_index=bi, hour_index=hi, right=right)
                self.assertFalse(at.exception, [e.value for e in at.exception])

    def test_metrics_match_saved_results(self):
        import json

        at = self.run_app("rot_oude_westen")
        res = json.loads((DASH / "rot_oude_westen" / "results.json").read_text())
        b = res["budgets"][2]
        want = next(r for r in res["rows"] if r["method"] == "nsga_gnn" and r["budget"] == b)
        self.assertEqual(at.metric[0].value, f"{want['engine_dose_reduction_pct']:.2f}%")


if __name__ == "__main__":
    unittest.main()
