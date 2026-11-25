"""Sky view factor (SVF) from horizon angles.

For a horizontal receptor and an isotropic sky, an azimuth sector whose horizon is
raised to elevation h contributes cos^2(h) of its unobstructed share, so
    SVF = (1/n) * sum_k cos^2(h_k)   (= 1 for open ground).
h_k is found by marching outwards along azimuth k and taking max atan(dz / distance).

Three variants are returned:
  svf_b     buildings only
  svf_bv    buildings + crowns treated as opaque
  svf       transmissivity-weighted: sky hidden only by crowns is partly visible,
            svf = svf_bv + (svf_b - svf_bv) * tree_transmissivity
"""
from __future__ import annotations

import math

import numpy as np

from .shadow import _steps, shift


def _horizon_svf(height: np.ndarray, base: np.ndarray, resolution_m: float, n_azimuths: int,
                 max_distance_m: float) -> np.ndarray:
    acc = np.zeros(base.shape, dtype=np.float64)
    hmax = float(height.max())
    for k in range(n_azimuths):
        az = 360.0 * k / n_azimuths
        tan_h = np.zeros(base.shape, dtype=np.float64)
        for _, dr, dc, dist in _steps(az, resolution_m, max_distance_m):
            if (hmax - base.min()) / dist < tan_h.min():
                break   # nothing farther can raise any pixel's horizon
            tan_h = np.maximum(tan_h, (shift(height, dr, dc) - base) / dist)
        acc += 1.0 / (1.0 + tan_h ** 2)     # cos^2(atan(x)) = 1 / (1 + x^2)
    return (acc / n_azimuths).astype(np.float32)


def compute(dsm, cdsm=None, resolution_m: float = 1.0, n_azimuths: int = 36,
            max_distance_m: float = 200.0, tree_transmissivity: float = 0.03) -> dict:
    """Return dict(svf, svf_b, svf_bv). Static per patch: cache the result (np.savez)."""
    dsm = np.asarray(dsm, dtype=np.float32)
    svf_b = _horizon_svf(dsm, dsm, resolution_m, n_azimuths, max_distance_m)
    if cdsm is None or not np.any(np.asarray(cdsm) > 0):
        return {"svf": svf_b, "svf_b": svf_b, "svf_bv": svf_b}
    top = np.maximum(dsm, np.asarray(cdsm, dtype=np.float32))
    # Crowns are treated as solid from the ground up to their top (trunk zone not
    # excluded): conservative for SVF, consistent with SOLWEIG's vegetation SVF.
    # A receptor's own crown is captured by the neighbouring crown pixels (crowns
    # are several metres wide at 1 m resolution).
    svf_bv = _horizon_svf(top, dsm, resolution_m, n_azimuths, max_distance_m)
    svf_bv = np.minimum(svf_bv, svf_b)
    svf = (svf_bv + (svf_b - svf_bv) * tree_transmissivity).astype(np.float32)
    return {"svf": svf, "svf_b": svf_b, "svf_bv": svf_bv}


def canyon_svf_analytic(height_m: float, width_m: float) -> float:
    """SVF at the centre of an infinitely long canyon floor (Oke): cos(atan(2H/W))."""
    return math.cos(math.atan(2.0 * height_m / width_m))
