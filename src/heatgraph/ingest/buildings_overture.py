"""Overture Maps buildings (global) via the official `overturemaps` Python client.

`height` is optional in the Overture schema, so missing heights are imputed from
`num_floors` (3.0 m per floor) and otherwise left NaN with height_source="missing".
The fraction of imputed/missing heights is written to the manifest: it is a key
covariate in the cross-dataset transfer experiment.
"""
from __future__ import annotations

from pathlib import Path

from . import Request

M_PER_FLOOR = 3.0


def plan(patch, ctx) -> list[Request]:
    from ..geo import approx_wgs84_bbox

    b = approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m)
    return [Request("overture", f"overturemaps building bbox={tuple(round(v, 5) for v in b)}",
                    est_mb=3.0, note="GeoParquet from Overture S3 release; heights often partial")]


def _geodataframe(bbox):
    try:
        from overturemaps import core
        return core.geodataframe("building", bbox=bbox)
    except (ImportError, AttributeError):
        from overturemaps import geodataframe
        return geodataframe("building", bbox=bbox)


def fetch(patch, ctx) -> list[Path]:
    import numpy as np
    from ..geo import approx_wgs84_bbox, working_crs

    out = ctx.patch_dir(patch)
    gdf = _geodataframe(approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m))
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    h = gdf["height"].astype(float) if "height" in gdf else np.nan
    floors = gdf["num_floors"].astype(float) if "num_floors" in gdf else np.nan
    gdf["height_m"] = np.where(np.isfinite(h), h, floors * M_PER_FLOOR)
    gdf["height_source"] = np.where(np.isfinite(h), "overture_height",
                                    np.where(np.isfinite(floors), "num_floors_x3", "missing"))
    keep = [c for c in ("id", "height_m", "height_source", "num_floors", "class", "geometry") if c in gdf]
    gdf = gdf[keep].to_crs(working_crs("overture", patch.lon, patch.lat))
    path = out / "buildings.gpkg"
    gdf.to_file(path, layer="buildings", driver="GPKG")
    return [path]
