"""LCZ-guided re-centring of flagged patches (3DBAG heights + Overture water).

    .\\.venv\\Scripts\\python.exe scripts/03_patch_search.py

For each flagged patch: fetch 3DBAG over a (2 x REGION_HALF) square around the current
centre (cached in data/cache/search/<patch>/), fetch Overture water, evaluate the 500 m
core metrics on a 50 m grid of candidate centres within SEARCH_RADIUS, and pick the best
feasible centre for the intended typology. Writes data/cache/search/patch_search.json
(all candidates + chosen centre in WGS84) so the choice is reproducible and auditable.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatgraph.config import DATA_DIR, load_patches  # noqa: E402
from heatgraph.geo import RD_NEW, approx_wgs84_bbox, projected_bbox  # noqa: E402
from heatgraph.ingest.buildings_3dbag import fetch_rd_bbox  # noqa: E402
from heatgraph.morphology import search_centre, window_metrics  # noqa: E402

REGION_HALF, SEARCH_RADIUS, STEP = 700.0, 450.0, 50.0


def high_rise(m):
    if m["water_frac"] > 0.15 or m["bsf"] < 0.20:
        return None
    return m["h_mean_aw"] + 1.0 * m["n_tall"]


def waterfront_mixed(m):
    # v1 rule (water 5-30 %, BSF >= 0.25) had 0/361 feasible windows: piers have water on several
    # sides and towers on small footprints. v2 (used): water <= 35 %, BSF >= 0.20, >= 3 towers,
    # with an explicit water penalty so the least-wet dense window wins.
    if m["water_frac"] > 0.35 or m["bsf"] < 0.20 or m["n_tall"] < 3:
        return None
    return 100 * m["bsf"] + m["h_mean_aw"] + 0.5 * m["n_tall"] - 50 * m["water_frac"]


TARGETS = {"rot_kop_van_zuid": waterfront_mixed, "dha_beatrixkwartier": high_rise, "ams_zuidas": high_rise}


def main():
    import geopandas as gpd
    from overturemaps import core
    from pyproj import Transformer

    _, patches = load_patches()
    to_wgs = Transformer.from_crs(RD_NEW, "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:4326", RD_NEW, always_xy=True)
    report = {}
    for pid, score in TARGETS.items():
        p = next(q for q in patches if q.id == pid)
        cache = DATA_DIR / "cache" / "search" / pid
        bpath = cache / "buildings_region.gpkg"
        if bpath.exists():
            b = gpd.read_file(bpath)
        else:
            b = fetch_rd_bbox(projected_bbox(p.lon, p.lat, REGION_HALF, RD_NEW), cache / "pages")
            b.to_file(bpath, driver="GPKG")
        w = core.geodataframe("water", bbox=approx_wgs84_bbox(p.lon, p.lat, REGION_HALF + 100))
        w = w.set_crs("EPSG:4326", allow_override=True)
        w = w[w.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(RD_NEW)
        c0 = to_rd.transform(p.lon, p.lat)
        old = window_metrics(b, w, *c0)
        best_xy, best_m, rows = search_centre(b, w, c0, SEARCH_RADIUS, STEP, score)
        lon, lat = to_wgs.transform(*best_xy) if best_xy else (None, None)
        report[pid] = {"old_centre": [p.lon, p.lat], "old_metrics": old,
                       "new_centre": [round(lon, 5), round(lat, 5)] if lon else None, "new_metrics": best_m,
                       "n_candidates": len(rows), "n_feasible": sum(r[2] is not None for r in rows)}
        print(pid, "| old", {k: round(v, 2) for k, v in old.items()}, "\n   new", report[pid]["new_centre"],
              {k: round(v, 2) for k, v in (best_m or {}).items()}, f"| feasible {report[pid]['n_feasible']}/{len(rows)}")
    (DATA_DIR / "cache" / "search" / "patch_search.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
