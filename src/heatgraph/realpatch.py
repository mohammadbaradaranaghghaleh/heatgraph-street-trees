"""Real patches: ingested files -> engine grids + forcing, SOLWEIG input export and comparison.

    p = load_patch("utr_binnenstad")          # grids on the 1 m working grid (RD New for NL)
    write_solweig_inputs(p, out_dir, day)     # Building_DSM.tif, DEM.tif, Trees.tif, metfile.txt
    m = compare_tmrt(ours, ref, mask)         # MAE / bias / RMSE, stratified

Conventions shared with SOLWEIG (checked in solweig-gpu 2.0.1 source):
  * Building_DSM = DEM + building height (absolute), DEM flat (0) inside a patch window;
    Trees = vegetation height ABOVE GROUND, 0 where none;
  * crown transmissivity 3 %, trunk zone 25 % of tree height (same as engine.toml);
  * met file = UMEP 24-column format in LOCAL time; radiation = preceding-hour mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_DIR, load_patches, load_toml
from .engine.surfaces import GridSpec, Grids, build_grids

MIN_TREE_HEIGHT_M = 2.0       # canopy below this is treated as grass/noise (Meta CHM is uint8 metres)
UMEP_HEADER = ("%iy id it imin qn qh qe qs qf U RH Tair press rain kdown snow ldown fcld wuh xsmd lai "
               "kdiff kdir wdir")


@dataclass
class RealPatch:
    patch: object
    grids: Grids
    spec: GridSpec
    crs: str
    weather: object            # DataFrame, UTC index
    core_mask: np.ndarray      # True inside the analysed 500 m core


def load_patch(patch_id: str, cfg: dict | None = None) -> RealPatch:
    import pandas as pd
    from .geo import projected_bbox, working_crs

    cfg = cfg or load_toml("engine.toml")
    meta, patches = load_patches()
    patch = next(p for p in patches if p.id == patch_id)
    d = DATA_DIR / "raw" / patch_id
    crs = working_crs(patch.source, patch.lon, patch.lat)
    half = meta.get("core_size_m", 500) / 2 + cfg["grid"]["buffer_m"]
    bounds = projected_bbox(patch.lon, patch.lat, half, crs)
    res = cfg["grid"]["resolution_m"]
    g = build_grids(d / "buildings.gpkg", bounds, res, canopy=d / "canopy_height_raw.tif", crs=crs,
                    trunk_zone_fraction=cfg["shadow"]["trunk_zone_fraction"])
    g.cdsm = np.where(g.cdsm >= MIN_TREE_HEIGHT_M, g.cdsm, 0.0).astype(np.float32)
    g.tdsm = (cfg["shadow"]["trunk_zone_fraction"] * g.cdsm).astype(np.float32)
    spec = g.transform
    w = pd.read_csv(d / "weather_hourly.csv", parse_dates=["time_utc"])
    w = w.set_index(pd.DatetimeIndex(w.pop("time_utc")).tz_localize("UTC"))
    core = np.zeros(g.dsm.shape, bool)
    b = int(round(cfg["grid"]["buffer_m"] / res))
    core[b:-b, b:-b] = True
    return RealPatch(patch, g, spec, crs, w, core)


def _write_tif(path, arr, spec: GridSpec, crs):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=from_origin(spec.minx, spec.maxy, spec.res, spec.res),
                       nodata=None) as dst:
        dst.write(arr.astype(np.float32), 1)


def local_day_rows(weather, day_local: str, tz: str = "Europe/Amsterdam"):
    """Rows of the UTC-indexed weather that fall on `day_local` in local time."""
    loc = weather.index.tz_convert(tz)
    return weather[loc.strftime("%Y-%m-%d") == day_local]


def umep_metfile(weather_rows, tz: str = "Europe/Amsterdam") -> str:
    """UMEP/SOLWEIG 24-column met file text (local time, -999 for unused columns)."""
    lines = [UMEP_HEADER]
    for ts, r in zip(weather_rows.index.tz_convert(tz), weather_rows.itertuples()):
        v = [-999.0] * 24
        v[0], v[1], v[2], v[3] = ts.year, ts.dayofyear, ts.hour, ts.minute
        v[9] = r.wind_speed_10m
        v[10] = r.relative_humidity_2m
        v[11] = r.temperature_2m
        v[12] = r.surface_pressure / 10.0                     # hPa -> kPa
        v[13] = 0.0                                           # rain
        v[14] = r.shortwave_radiation                         # global horizontal
        v[21] = r.diffuse_radiation                           # diffuse horizontal
        v[22] = r.direct_normal_irradiance                    # direct (beam-normal)
        lines.append(" ".join(f"{int(x)}" if i < 4 else f"{x:.2f}" for i, x in enumerate(v)))
    return "\n".join(lines) + "\n"


def write_solweig_inputs(p: RealPatch, out_dir: Path, day_local: str, cdsm=None) -> dict:
    """cdsm: optional canopy-height grid (e.g. existing canopy + planted crowns); default p.grids.cdsm."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = p.grids
    dem = np.zeros_like(g.dsm)
    _write_tif(out_dir / "Building_DSM.tif", dem + g.dsm, p.spec, p.crs)
    _write_tif(out_dir / "DEM.tif", dem, p.spec, p.crs)
    _write_tif(out_dir / "Trees.tif", g.cdsm if cdsm is None else np.asarray(cdsm, np.float32), p.spec, p.crs)
    rows = local_day_rows(p.weather, day_local)
    (out_dir / "metfile.txt").write_text(umep_metfile(rows), encoding="utf-8")
    info = {"patch": p.patch.id, "day_local": day_local, "crs": p.crs, "shape": list(g.dsm.shape),
            "bounds": list(p.spec.bounds), "hours_local": [t.strftime("%H:%M") for t in
                                                         rows.index.tz_convert("Europe/Amsterdam")]}
    (out_dir / "inputs.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def compare_tmrt(ours, ref, mask, sunlit=None, under_canopy=None) -> dict:
    """Pixel-hour Tmrt agreement on `mask` (ground pixels in the core). Arrays (hours, H, W)."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    m = np.broadcast_to(mask, ours.shape) & np.isfinite(ours) & np.isfinite(ref)
    d = (ours - ref)[m]
    out = {"n": int(m.sum()), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d)),
           "rmse": float(np.sqrt(np.mean(d ** 2))), "p95_abs": float(np.percentile(np.abs(d), 95))}
    strata = {}
    if sunlit is not None:
        sl = np.asarray(sunlit)
        strata["sunlit"] = m & (sl > 0.5)
        strata["shaded"] = m & (sl <= 0.5)
    if under_canopy is not None:
        strata["under_canopy"] = m & np.broadcast_to(under_canopy, ours.shape)
    for k, sm in strata.items():
        dd = (ours - ref)[sm]
        if dd.size:
            out[k] = {"n": int(sm.sum()), "mae": float(np.mean(np.abs(dd))), "bias": float(np.mean(dd))}
    return out
