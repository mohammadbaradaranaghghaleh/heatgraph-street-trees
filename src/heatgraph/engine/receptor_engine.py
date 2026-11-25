"""Receptor-level, batched, GPU-capable version of the Block C engine (label generator).

Same equations and step geometry as shadow.cast / svf.compute / radiation.tmrt /
comfort.utci_grid, but evaluated only at pedestrian receptor pixels and for a batch of
plantings at once (torch, CUDA if available). Exactness arguments:

  * shifted-array ops restricted to a pixel equal a gather at (pixel + offset) with the
    same out-of-grid fill (0), so per-receptor results are identical to the full-grid engine;
  * horizon over max(dsm, cdsm) = max(horizon over dsm, horizon over cdsm) per azimuth, so the
    building horizon is computed once and only the canopy part is recomputed per planting;
  * building shadow does not depend on trees, so it is computed once per hour.

Equivalence with the full-grid numpy engine is asserted in tests/test_receptor_engine.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import comfort, radiation, sun
from .shadow import _steps


class ReceptorEngine:
    def __init__(self, dsm, receptors_rc, weather, lat: float, lon: float, cfg: dict, device: str | None = None,
                 resolution_m: float = 1.0):
        import pandas as pd

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg, self.res = cfg, resolution_m
        self.trans = cfg["shadow"]["tree_transmissivity"]
        self.trunk = cfg["shadow"].get("trunk_zone_fraction", 0.25)
        self.dsm_np = np.asarray(dsm, np.float32)
        self.H, self.W = self.dsm_np.shape
        self.dsm = torch.from_numpy(self.dsm_np).to(self.device)
        rc = np.asarray(receptors_rc, np.int64)
        self.rc = torch.from_numpy(rc).to(self.device)
        self.base = self.dsm[self.rc[:, 0], self.rc[:, 1]]                  # receptor surface height
        self.weather = weather

        off = pd.Timedelta(minutes=cfg["sun"].get("time_offset_min", 0))
        pos = sun.positions(weather.index + off, lat, lon)
        self.elev, self.az = pos["elevation"].to_numpy(), pos["azimuth"].to_numpy()
        self.dni, self.dhi, _ = sun.split_radiation(weather["shortwave_radiation"].to_numpy(), pos["zenith"].to_numpy(),
                                                    weather.index + off, weather.get("direct_normal_irradiance"),
                                                    weather.get("diffuse_radiation"))
        self.min_el = cfg["sun"]["min_elevation_deg"]
        self.maxd_shadow = cfg["grid"]["buffer_m"]
        self.n_az, self.maxd_svf = cfg["svf"]["n_azimuths"], cfg["svf"]["max_distance_m"]
        self.az_steps = [list(_steps(360.0 * k / self.n_az, self.res, self.maxd_svf)) for k in range(self.n_az)]
        self.sun_steps = [list(_steps(self.az[h], self.res, self.maxd_shadow)) if self.elev[h] > self.min_el else None
                          for h in range(len(weather))]
        rad = cfg.get("radiation", {})
        self.params = {k: v for k, v in rad.items() if not isinstance(v, str)}

        # ---- planting-independent precomputation
        with torch.no_grad():
            self.tan_b = torch.stack([self._horizon(self.dsm[None], steps)[0] for steps in self.az_steps])  # (n_az, N)
            self.svf_b = (1.0 / (1.0 + self.tan_b ** 2)).mean(0)                                         # (N,)
            self.bshadow = torch.stack([self._bshadow(h) for h in range(len(weather))])                  # (hours, N)

    # ------------------------------------------------------------------ gathers
    def _gather(self, grid, dr: int, dc: int):
        """grid (B, H, W) -> (B, N) values at receptor + (dr, dc); 0 outside the grid."""
        r, c = self.rc[:, 0] + dr, self.rc[:, 1] + dc
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        vals = grid[:, r.clamp(0, self.H - 1), c.clamp(0, self.W - 1)]
        return torch.where(ok[None], vals, torch.zeros((), device=grid.device, dtype=grid.dtype))

    def _horizon(self, height, steps):
        """max over steps of (height(q) - base) / dist, floored at 0 (svf._horizon_svf logic)."""
        tan = torch.zeros((height.shape[0], self.rc.shape[0]), device=self.device)
        for _, dr, dc, dist in steps:
            tan = torch.maximum(tan, (self._gather(height, dr, dc) - self.base[None]) / dist)
        return tan

    def _bshadow(self, h: int):
        if self.sun_steps[h] is None:
            return torch.ones(self.rc.shape[0], dtype=torch.bool, device=self.device)
        tan_e = math.tan(math.radians(self.elev[h]))
        hmax = float(self.dsm.max())
        sh = torch.zeros(self.rc.shape[0], dtype=torch.bool, device=self.device)
        for _, dr, dc, dist in self.sun_steps[h]:
            rise = dist * tan_e
            if rise > hmax:          # numpy: break when every pixel's ray is above all surfaces
                break
            sh |= self._gather(self.dsm[None], dr, dc)[0] > self.base + rise + 1e-6
        return sh

    # ------------------------------------------------------------------ per planting
    @torch.no_grad()
    def sky_and_sun(self, cdsm_batch):
        """cdsm_batch (B, H, W) canopy heights (0 under buildings) -> svf (B, N), sunlit (B, hours, N)."""
        c = torch.as_tensor(np.asarray(cdsm_batch, np.float32), device=self.device)
        t = self.trunk * c
        B = c.shape[0]
        acc = torch.zeros((B, self.rc.shape[0]), device=self.device)
        any_veg = c.reshape(B, -1).amax(1) > 0                              # numpy returns svf_b if no canopy
        for k, steps in enumerate(self.az_steps):
            tan = torch.maximum(self.tan_b[k][None], self._horizon(c, steps))
            acc += 1.0 / (1.0 + tan ** 2)
        svf_bv = torch.minimum(acc / self.n_az, self.svf_b[None])
        svf = svf_bv + (self.svf_b[None] - svf_bv) * self.trans
        svf = torch.where(any_veg[:, None], svf, self.svf_b[None].expand(B, -1))

        hmax_c = c.reshape(B, -1).amax(1)
        hmax = torch.maximum(hmax_c, self.dsm.max())
        sunlit = torch.zeros((B, len(self.weather), self.rc.shape[0]), device=self.device)
        for h, steps in enumerate(self.sun_steps):
            if steps is None:
                continue                                                    # night / sun too low: 0
            tan_e = math.tan(math.radians(self.elev[h]))
            v = torch.zeros((B, self.rc.shape[0]), dtype=torch.bool, device=self.device)
            for _, dr, dc, dist in steps:
                rise = dist * tan_e
                if rise > float(hmax.max()):
                    break
                z = self.base[None] + rise
                live = (rise <= hmax)[:, None]                              # per-planting early stop
                cc, tt = self._gather(c, dr, dc), self._gather(t, dr, dc)
                v |= live & (cc > 0) & (z < cc) & (z > tt)
            b = self.bshadow[h][None].expand(B, -1)
            sunlit[:, h] = torch.where(b, 0.0, torch.where(v, self.trans, 1.0))
        return svf, sunlit

    def utci(self, cdsm_batch):
        """-> dict(tmrt, utci) each (B, hours, N) numpy, plus svf (B, N) and sunlit."""
        svf, sunlit = self.sky_and_sun(cdsm_batch)
        svf_np, sl_np = svf.cpu().numpy(), sunlit.cpu().numpy()
        B, Hh, N = sl_np.shape
        tm = np.empty((B, Hh, N), np.float32)
        ut = np.empty((B, Hh, N), np.float32)
        w = self.weather
        red = self.cfg["wind"].get("urban_reduction_factor", 1.0)
        for h in range(Hh):
            ta, rh = float(w["temperature_2m"].iloc[h]), float(w["relative_humidity_2m"].iloc[h])
            tm[:, h] = radiation.tmrt(sl_np[:, h], svf_np, ta, rh, self.dni[h], self.dhi[h], self.elev[h], self.params)
            ut[:, h] = comfort.utci_grid(ta, tm[:, h], float(w["wind_speed_10m"].iloc[h]) * red, rh)
        return {"tmrt": tm, "utci": ut, "svf": svf_np, "sunlit": sl_np}


def segment_mean(values, seg_of_receptor, n_segments: int):
    """(..., N) receptor values -> (..., n_segments) per-segment means."""
    v = np.asarray(values, np.float64)
    seg = np.asarray(seg_of_receptor)
    cnt = np.bincount(seg, minlength=n_segments).astype(np.float64)
    flat = v.reshape(-1, v.shape[-1])
    out = np.stack([np.bincount(seg, weights=row, minlength=n_segments) for row in flat]) / cnt
    return out.reshape(*v.shape[:-1], n_segments).astype(np.float32)
