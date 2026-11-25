"""Block C driver: hourly Tmrt/UTCI for one patch.

    out = simulate(grids, weather, lat, lon, cfg)       # cfg = configs/engine.toml as dict

`weather` is a pandas DataFrame indexed by tz-aware UTC timestamps with columns
temperature_2m (C), relative_humidity_2m (%), wind_speed_10m (m/s),
shortwave_radiation (GHI) and optionally direct_normal_irradiance / diffuse_radiation.
SVF is computed once per patch; shadows, Tmrt and UTCI per hour.
"""
from __future__ import annotations

import numpy as np

from . import comfort, radiation, shadow, sun, svf as svf_mod
from .surfaces import Grids


def simulate(grids: Grids, weather, lat: float, lon: float, cfg: dict, svf_cache: dict | None = None,
             hours: list | None = None) -> dict:
    sh, sv, wd, rad = cfg["shadow"], cfg["svf"], cfg["wind"], cfg.get("radiation", {})
    if hours is not None:
        weather = weather.loc[hours]
    sky = svf_cache or svf_mod.compute(grids.dsm, grids.cdsm, grids.resolution_m, sv["n_azimuths"],
                                       sv["max_distance_m"], sh["tree_transmissivity"])
    # Radiation given as the mean of the PRECEDING interval (Open-Meteo/ERA5 accumulations) is
    # evaluated with the sun at the interval midpoint, as SOLWEIG does: time_offset_min = -30.
    import pandas as pd

    offset = pd.Timedelta(minutes=cfg["sun"].get("time_offset_min", 0))
    pos = sun.positions(weather.index + offset, lat, lon)
    pos.index = weather.index
    dni, dhi, _ = sun.split_radiation(weather["shortwave_radiation"].to_numpy(), pos["zenith"].to_numpy(),
                                      weather.index + offset, weather.get("direct_normal_irradiance"),
                                      weather.get("diffuse_radiation"))
    params = {k: rad[k] for k in ("albedo_ground", "albedo_wall", "emissivity_ground", "emissivity_wall",
                                  "absorption_shortwave", "absorption_longwave", "ground_warming_k_per_kwm2",
                                  "wall_warming_k_per_kwm2") if k in rad}

    n, shape = len(weather), grids.dsm.shape
    tm = np.empty((n, *shape), dtype=np.float32)
    ut = np.empty((n, *shape), dtype=np.float32)
    sl = np.empty((n, *shape), dtype=np.float32)
    for i, (ts, row) in enumerate(weather.iterrows()):
        el, az = float(pos["elevation"].iloc[i]), float(pos["azimuth"].iloc[i])
        if el > cfg["sun"]["min_elevation_deg"]:
            sunlit, _, _ = shadow.cast(grids.dsm, grids.cdsm, grids.tdsm, az, el, grids.resolution_m,
                                       cfg["grid"]["buffer_m"], sh["tree_transmissivity"])
        else:
            sunlit = np.zeros(shape, dtype=np.float32)
        ta, rh = float(row["temperature_2m"]), float(row["relative_humidity_2m"])
        tm[i] = radiation.tmrt(sunlit, sky["svf"], ta, rh, dni[i], dhi[i], el, params)
        v = float(row["wind_speed_10m"]) * wd.get("urban_reduction_factor", 1.0)
        ut[i] = comfort.utci_grid(ta, tm[i], v, rh)
        sl[i] = sunlit
    return {"times": weather.index, "sunlit": sl, "tmrt": tm, "utci": ut, **sky,
            "elevation": pos["elevation"].to_numpy(), "azimuth": pos["azimuth"].to_numpy()}
