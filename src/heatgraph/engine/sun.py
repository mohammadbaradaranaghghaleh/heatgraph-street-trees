"""Solar geometry (pvlib, NREL SPA) and beam/diffuse handling of the forcing.

Grid convention used throughout the engine: row 0 is the northern edge, rows increase
southwards, columns increase eastwards; azimuth is degrees clockwise from north.
"""
from __future__ import annotations

import numpy as np


def positions(times_utc, lat: float, lon: float, altitude_m: float = 0.0):
    """DataFrame with columns elevation, azimuth, zenith (deg) indexed by time.

    `times_utc` must be tz-aware (or a DatetimeIndex localised to UTC).
    Uses apparent (refraction-corrected) elevation, as appropriate for shading.
    """
    import pandas as pd
    import pvlib

    times = pd.DatetimeIndex(times_utc)
    if times.tz is None:
        times = times.tz_localize("UTC")
    sp = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=altitude_m, method="nrel_numpy")
    return pd.DataFrame({"elevation": sp["apparent_elevation"], "azimuth": sp["azimuth"],
                         "zenith": sp["apparent_zenith"]}, index=times)


def split_radiation(ghi, zenith_deg, times_utc=None, dni=None, dhi=None):
    """Return (dni, dhi, ghi) in W/m2 as float arrays.

    If the forcing provides DNI and DHI they are used directly; otherwise GHI is
    decomposed with the Erbs et al. (1982) model via pvlib.irradiance.erbs.
    DNI is set to 0 when the sun is below the horizon.
    """
    ghi = np.atleast_1d(np.asarray(ghi, dtype=float))
    zen = np.atleast_1d(np.asarray(zenith_deg, dtype=float))
    if dni is not None and dhi is not None:
        dni_a = np.atleast_1d(np.asarray(dni, dtype=float))
        dhi_a = np.atleast_1d(np.asarray(dhi, dtype=float))
    else:
        import pandas as pd
        import pvlib

        if times_utc is None:
            raise ValueError("times_utc is required for the Erbs decomposition")
        doy = np.asarray(pd.DatetimeIndex(times_utc).day_of_year)
        out = pvlib.irradiance.erbs(ghi, zen, doy)
        dni_a, dhi_a = np.asarray(out["dni"], float), np.asarray(out["dhi"], float)
    below = zen >= 90.0
    dni_a = np.where(below | ~np.isfinite(dni_a), 0.0, np.maximum(dni_a, 0.0))
    dhi_a = np.where(~np.isfinite(dhi_a), 0.0, np.maximum(dhi_a, 0.0))
    return dni_a, dhi_a, ghi


def sun_vector_grid(azimuth_deg: float):
    """Unit step (d_row, d_col) pointing from a pixel TOWARDS the sun on the grid."""
    a = np.radians(azimuth_deg)
    return -np.cos(a), np.sin(a)   # north = -row, east = +col
