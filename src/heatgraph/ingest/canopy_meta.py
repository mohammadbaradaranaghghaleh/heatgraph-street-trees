"""Meta/WRI 1 m global canopy height (CHM v1; Tolan et al., 2024) from AWS Open Data.

Public bucket, no credentials: s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/
The tile index `tiles.geojson` (~15 MB, size checked 2026-11-24) is downloaded ONCE
and cached under data/cache/. Tiles are Cloud-Optimised GeoTIFFs read with a
windowed request, so only the patch window is transferred, not the whole tile.

Tile-id field VERIFIED 2026-11-25 against the cached index: 56,145 polygons (WGS84), one
property `tile` holding a 9-digit zoom-9 Bing quadkey (e.g. "023013213"); the COG for a
tile is chm/<tile>.tif.
A CHM v2 release also exists (AWS registry: dataforgood-fb-forestsv2); keep v1 for
consistency with the cited paper unless v2 is cited and validated.
"""
from __future__ import annotations

from pathlib import Path

from . import Request

BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
INDEX_URL = f"{BASE}/tiles.geojson"
TILE_URL = BASE + "/chm/{tile}.tif"
TILE_ID_FIELDS = ("tile",)   # verified; anything else means the index format changed -> fail loudly


def plan(patch, ctx) -> list[Request]:
    reqs = []
    if not (ctx.cache_dir / "meta_chm_tiles.geojson").exists() and INDEX_URL not in ctx.planned:
        ctx.planned.add(INDEX_URL)
        reqs.append(Request("meta_chm", INDEX_URL, est_mb=15.2, note="tile index, cached once for all patches"))
    side = 2 * ctx.half_size_m
    est = side * side * 4 / 1e6  # float32 window, before compression
    reqs.append(Request("meta_chm", TILE_URL.format(tile="<tile id from index>"), est_mb=round(est, 1),
                        note="windowed COG read via /vsicurl/"))
    return reqs


def _index(ctx):
    import urllib.request
    import geopandas as gpd

    p = ctx.cache_dir / "meta_chm_tiles.geojson"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(INDEX_URL, p)
    return gpd.read_file(p)


def fetch(patch, ctx) -> list[Path]:
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from ..geo import approx_wgs84_bbox

    out = ctx.patch_dir(patch)
    b = approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m)
    idx = _index(ctx)
    field = next((f for f in TILE_ID_FIELDS if f in idx.columns), None)
    if field is None:
        raise KeyError(f"No tile-id field among {TILE_ID_FIELDS}; columns are {list(idx.columns)}")
    hits = idx[idx.intersects(box(*b))]
    if hits.empty:
        raise ValueError(f"No CHM tile covers patch {patch.id}")

    srcs = [rasterio.open(f"/vsicurl/{TILE_URL.format(tile=t)}") for t in hits[field]]
    try:
        bounds = transform_bounds("EPSG:4326", srcs[0].crs, *b)
        mosaic, transform = merge(srcs, bounds=bounds)
        src0 = srcs[0]
        # fresh profile: inheriting the COG's block layout breaks on arbitrary window sizes
        profile = {"driver": "GTiff", "dtype": mosaic.dtype, "count": mosaic.shape[0],
                   "height": mosaic.shape[1], "width": mosaic.shape[2], "crs": src0.crs,
                   "transform": transform, "nodata": src0.nodata, "compress": "deflate",
                   "tiled": True, "blockxsize": 256, "blockysize": 256}
    finally:
        for s in srcs:
            s.close()
    path = out / "canopy_height_raw.tif"   # native CRS; reprojected to the working grid in Block B
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mosaic)
    except Exception:
        path.unlink(missing_ok=True)       # never leave a truncated file for the manifest to hash
        raise
    return [path]
