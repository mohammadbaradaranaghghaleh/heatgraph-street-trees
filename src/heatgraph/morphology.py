"""Urban-form metrics for patch QA and LCZ-guided patch placement (Stewart & Oke, 2012).

Computed on the 500 m CORE of a patch from building footprints (3DBAG heights) and water
polygons. Used to check that each patch matches its intended Local Climate Zone and to
search for better patch centres:

  bsf        building surface fraction (footprint area / core area)
  h_mean_aw  area-weighted mean building height (m)
  h_p90      90th percentile building height (m), footprint-area weighted
  n_tall     buildings taller than 40 m
  water_frac water area / core area
"""
from __future__ import annotations

import numpy as np


def window_metrics(buildings, water, cx: float, cy: float, half: float = 250.0, tall_m: float = 40.0) -> dict:
    """buildings: GeoDataFrame with height_m (projected CRS); water: GeoDataFrame or None."""
    from shapely.geometry import box

    win = box(cx - half, cy - half, cx + half, cy + half)
    area = win.area
    b = buildings[buildings.intersects(win)]
    clip = b.geometry.intersection(win)
    a = clip.area.to_numpy()
    h = b["height_m"].to_numpy(dtype=float)
    ok = np.isfinite(h) & (a > 0)
    out = {"bsf": float(a.sum() / area), "n_buildings": int(len(b)), "n_tall": int(np.sum(h[ok] > tall_m))}
    if ok.any():
        w = a[ok] / a[ok].sum()
        order = np.argsort(h[ok])
        cum = np.cumsum(w[order])
        out["h_mean_aw"] = float(np.sum(w * h[ok]))
        out["h_p90"] = float(h[ok][order][np.searchsorted(cum, 0.9)])
    else:
        out["h_mean_aw"] = out["h_p90"] = 0.0
    if water is not None and len(water):
        wv = water[water.intersects(win)]
        out["water_frac"] = float(wv.geometry.intersection(win).union_all().area / area) if len(wv) else 0.0
    else:
        out["water_frac"] = 0.0
    return out


def search_centre(buildings, water, centre_xy, radius: float, step: float, score_fn, half: float = 250.0):
    """Evaluate window_metrics on a grid of candidate centres within `radius` of centre_xy;
    return (best_xy, best_metrics, all_rows) maximising score_fn(metrics) (None = infeasible)."""
    cx0, cy0 = centre_xy
    rows = []
    offs = np.arange(-radius, radius + 1e-6, step)
    for dx in offs:
        for dy in offs:
            m = window_metrics(buildings, water, cx0 + dx, cy0 + dy, half)
            s = score_fn(m)
            rows.append((cx0 + dx, cy0 + dy, s, m))
    feas = [r for r in rows if r[2] is not None]
    if not feas:
        return None, None, rows
    best = max(feas, key=lambda r: r[2])
    return (best[0], best[1]), best[3], rows
