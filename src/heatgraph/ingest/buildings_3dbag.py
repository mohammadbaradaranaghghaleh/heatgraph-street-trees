"""3DBAG buildings (Netherlands) via the 3DBAG OGC API - Features endpoint.

Endpoint checked on 2026-11-24: GET https://api.3dbag.nl/collections/pand/items?bbox=...
returns CityJSONFeature objects; bbox is in RD New (EPSG:28992).
Height attributes used (3DBAG naming, checked against a live response):
  b3_h_maaiveld   ground level (m, NAP)
  b3_h_dak_70p    70th-percentile roof height (m, NAP)  -> main height proxy
  b3_h_dak_max    maximum roof height (m, NAP)
  oorspronkelijkbouwjaar  construction year
Raw pages are always saved, so footprints can be re-parsed if the parser changes.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import Request

API = "https://api.3dbag.nl/collections/pand/items"
PAGE_LIMIT = 100
USER_AGENT = "HeatGraph-research/0.1 (academic data ingestion)"


def _url(bbox_rd: tuple[float, float, float, float] | None) -> str:
    if bbox_rd is None:
        return f"{API}?bbox=<RD bbox computed at execution>&limit={PAGE_LIMIT}"
    b = ",".join(f"{v:.1f}" for v in bbox_rd)
    return f"{API}?{urllib.parse.urlencode({'bbox': b, 'limit': PAGE_LIMIT})}"


def plan(patch, ctx) -> list[Request]:
    # ~900 m x 900 m (core + buffer) in a Dutch city: roughly 1-4k buildings, ~1 MB per 100.
    side_km = 2 * ctx.half_size_m / 1000
    return [Request("3dbag", _url(None), est_mb=round(25 * side_km**2, 1),
                    note="paged CityJSONFeature responses; heights from b3_* attributes")]


def _get_json(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _transform(page: dict, feature: dict):
    for holder in (feature, page, page.get("metadata", {})):
        t = holder.get("transform") if isinstance(holder, dict) else None
        if t:
            return t["scale"], t["translate"]
    return None


def parse_features(page: dict) -> list[dict]:
    """Extract LoD0 footprints + height attributes from one API page."""
    rows = []
    for feat in page.get("features", []):
        verts = feat.get("vertices", [])
        tr = _transform(page, feat)

        def xy(i):
            v = verts[i]
            if tr:
                (sx, sy, _), (tx, ty, _) = tr
                return (v[0] * sx + tx, v[1] * sy + ty)
            return (v[0], v[1])

        for oid, obj in feat.get("CityObjects", {}).items():
            if obj.get("type") != "Building":
                continue
            a = obj.get("attributes", {})
            lod0 = [g for g in obj.get("geometry", []) if str(g.get("lod")) == "0"]
            if not lod0:
                continue
            # MultiSurface -> first surface -> outer ring (+ holes)
            surface = lod0[0]["boundaries"][0]
            rings = [[xy(i) for i in ring] for ring in surface]
            ground, roof70 = a.get("b3_h_maaiveld"), a.get("b3_h_dak_70p")
            height = (roof70 - ground) if (roof70 is not None and ground is not None) else None
            rows.append({
                "id": oid,
                "rings": rings,
                "height_m": height,
                "h_ground": ground,
                "h_roof_70p": roof70,
                "h_roof_max": a.get("b3_h_dak_max"),
                "roof_type": a.get("b3_dak_type"),
                "year_built": a.get("oorspronkelijkbouwjaar"),
                "height_source": "3dbag_b3_h_dak_70p" if height is not None else "missing",
            })
    return rows


def fetch_rd_bbox(bbox_rd, raw_dir: Path):
    """All 3DBAG buildings in an RD New bbox -> GeoDataFrame (raw pages saved to raw_dir)."""
    from shapely.geometry import Polygon
    import geopandas as gpd
    from ..geo import RD_NEW

    raw_dir.mkdir(parents=True, exist_ok=True)
    url = _url(bbox_rd)
    rows, page_no = [], 0
    while url:
        page = _get_json(url)
        (raw_dir / f"page_{page_no:04d}.json").write_text(json.dumps(page), encoding="utf-8")
        rows.extend(parse_features(page))
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
        page_no += 1
        time.sleep(0.2)
    geoms = [Polygon(r["rings"][0], r["rings"][1:]) for r in rows]
    gdf = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "rings"} for r in rows], geometry=geoms, crs=RD_NEW)
    return gdf.drop_duplicates("id")


def fetch(patch, ctx) -> list[Path]:
    from shapely.geometry import Polygon          # lazy heavy imports
    import geopandas as gpd
    from ..geo import RD_NEW, projected_bbox

    out = ctx.patch_dir(patch)
    raw_dir = out / "3dbag_pages"
    if raw_dir.exists():                          # stale pages from a previous centre must not linger
        for f in raw_dir.glob("page_*.json"):
            f.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = _url(projected_bbox(patch.lon, patch.lat, ctx.half_size_m, RD_NEW))
    rows, page_no = [], 0
    while url:
        page = _get_json(url)
        (raw_dir / f"page_{page_no:04d}.json").write_text(json.dumps(page), encoding="utf-8")
        rows.extend(parse_features(page))
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
        page_no += 1
        time.sleep(0.2)  # be polite to a free public API

    geoms = [Polygon(r["rings"][0], r["rings"][1:]) for r in rows]
    gdf = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "rings"} for r in rows],
                           geometry=geoms, crs=RD_NEW)
    gdf = gdf.drop_duplicates("id")
    path = out / "buildings.gpkg"
    gdf.to_file(path, layer="buildings", driver="GPKG")
    return [path, raw_dir]
