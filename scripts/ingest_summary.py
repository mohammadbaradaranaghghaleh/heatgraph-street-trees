"""Summarise ingested patches (data/raw/*): sizes, height completeness, canopy, network.

    .\\.venv\\Scripts\\python.exe scripts/ingest_summary.py   -> prints a table, writes data/raw/ingest_summary.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))


def main():
    import geopandas as gpd
    import rasterio

    out = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        m = d / "manifest.json"
        if not m.exists():
            continue
        j = json.loads(m.read_text(encoding="utf-8"))
        row = {"patch": d.name, "source": j["patch"]["source"], "errors": ";".join(j["errors"]),
               "mb": round(sum(v["bytes"] for v in j["files"].values()) / 1e6, 1)}
        if (d / "buildings.gpkg").exists():
            b = gpd.read_file(d / "buildings.gpkg")
            row.update(buildings=len(b), height_missing_pct=round(100 * float(b["height_m"].isna().mean()), 1),
                       height_median_m=round(float(b["height_m"].median()), 1))
            # LCZ form metrics on the 500 m core (heatgraph.morphology)
            from heatgraph.geo import projected_bbox, working_crs
            from heatgraph.morphology import window_metrics

            p = j["patch"]
            crs = working_crs(p["source"], p["lon"], p["lat"])
            x0, y0, x1, y1 = projected_bbox(p["lon"], p["lat"], 1.0, crs)
            water = None
            if (d / "network.gpkg").exists():
                try:
                    water = gpd.read_file(d / "network.gpkg", layer="water").to_crs(crs)
                except Exception:
                    water = None
            m = window_metrics(b.to_crs(crs), water, (x0 + x1) / 2, (y0 + y1) / 2)
            row.update(core_bsf=round(m["bsf"], 2), core_h_mean_aw=round(m["h_mean_aw"], 1),
                       core_n_tall=m["n_tall"], core_water_frac=round(m["water_frac"], 2))
        if (d / "canopy_height_raw.tif").exists():
            with rasterio.open(d / "canopy_height_raw.tif") as s:
                row["canopy_gt2m_pct"] = round(100 * float((s.read(1) > 2).mean()), 1)
        if (d / "network.gpkg").exists():
            e = gpd.read_file(d / "network.gpkg", layer="edges")
            row.update(walk_km=round(e["length_m"].sum() / 1000, 1),
                       destinations=len(gpd.read_file(d / "network.gpkg", layer="destinations")))
        out.append(row)
    keys = ["patch", "source", "buildings", "height_missing_pct", "height_median_m", "core_bsf", "core_h_mean_aw",
            "core_n_tall", "core_water_frac", "canopy_gt2m_pct", "walk_km", "destinations", "mb", "errors"]
    print(" | ".join(keys))
    for r in out:
        print(" | ".join(str(r.get(k, "")) for k in keys))
    with open(RAW / "ingest_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in keys} for r in out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
