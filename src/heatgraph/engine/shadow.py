"""2.5D raster shadow casting (shifted-max method of Ratti & Richens, 2004), numpy-vectorised.

From every pixel p a ray is traced towards the sun. After s steps (step length L metres
along the ground) the ray is at height z(s) = H(p) + s * L * tan(e). The pixel is
  * building-shaded  if any building/terrain surface on the path exceeds z(s);
  * vegetation-shaded if the ray passes through a crown, i.e. tdsm(q) < z(s) < cdsm(q).
Rather than tracing per pixel, whole arrays are shifted one step at a time, so the cost
is O(n_steps * n_pixels). Cells outside the grid are treated as open ground, which is
why the ingestion window carries a buffer around the analysed core.
"""
from __future__ import annotations

import math

import numpy as np

from .sun import sun_vector_grid


def shift(a: np.ndarray, dr: int, dc: int, fill: float = 0.0) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc] (value found dr rows / dc cols away), `fill` outside."""
    out = np.full_like(a, fill)
    H, W = a.shape
    r0, r1 = max(0, -dr), min(H, H - dr)
    c0, c1 = max(0, -dc), min(W, W - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def _steps(azimuth_deg: float, resolution_m: float, max_distance_m: float):
    """Yield (s, dr, dc, distance_m) for successive pixel offsets towards the sun."""
    drow, dcol = sun_vector_grid(azimuth_deg)
    scale = 1.0 / max(abs(drow), abs(dcol))     # advance one pixel along the dominant axis
    step_m = resolution_m * scale * math.hypot(drow, dcol)
    n = int(math.ceil(max_distance_m / step_m))
    last = None
    for s in range(1, n + 1):
        dr, dc = int(round(s * drow * scale)), int(round(s * dcol * scale))
        if (dr, dc) == last:
            continue
        last = (dr, dc)
        yield s, dr, dc, s * step_m


def cast(dsm, cdsm, tdsm, azimuth_deg: float, elevation_deg: float, resolution_m: float = 1.0,
         max_distance_m: float = 200.0, tree_transmissivity: float = 0.03):
    """Return (sunlit_fraction, building_shadow, vegetation_shadow).

    building_shadow / vegetation_shadow are bool arrays; sunlit_fraction is 1 in sun,
    `tree_transmissivity` under vegetation shade only, 0 in building shade (and at night).
    """
    dsm = np.asarray(dsm, dtype=np.float32)
    if elevation_deg <= 0:
        z = np.zeros_like(dsm, dtype=bool)
        return np.zeros_like(dsm), ~z, z
    has_veg = cdsm is not None and np.any(cdsm > 0)
    cdsm = np.zeros_like(dsm) if cdsm is None else np.asarray(cdsm, dtype=np.float32)
    tdsm = np.zeros_like(dsm) if tdsm is None else np.asarray(tdsm, dtype=np.float32)

    tan_e = math.tan(math.radians(elevation_deg))
    hmax = float(max(dsm.max(), cdsm.max()))
    bshadow = np.zeros(dsm.shape, dtype=bool)
    vshadow = np.zeros(dsm.shape, dtype=bool)
    for _, dr, dc, dist in _steps(azimuth_deg, resolution_m, max_distance_m):
        rise = dist * tan_e
        z = dsm + rise                      # ray height above ground at this offset
        if np.all(z > hmax):
            break                           # ray is above every obstacle in the scene
        bshadow |= shift(dsm, dr, dc) > z + 1e-6
        if has_veg:
            c, t = shift(cdsm, dr, dc), shift(tdsm, dr, dc)
            vshadow |= (c > 0) & (z < c) & (z > t)
    sunlit = np.where(bshadow, 0.0, np.where(vshadow, tree_transmissivity, 1.0)).astype(np.float32)
    return sunlit, bshadow, vshadow & ~bshadow
