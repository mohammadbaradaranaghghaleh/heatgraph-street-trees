"""UTCI via pythermalcomfort (Bröde et al., 2012 operational procedure)."""
from __future__ import annotations

import warnings

import numpy as np

V_MIN, V_MAX = 0.5, 17.0   # validity range of the UTCI regression (m/s at 10 m)


def wind_at_height(v_ref, z_ref: float = 10.0, z: float = 1.1, z0: float = 0.5):
    """Neutral log-profile wind speed at height z from a reference height (m/s).

    Diagnostic only: UTCI itself takes a 10 m(-equivalent) wind, see utci_grid.
    """
    return np.asarray(v_ref, dtype=float) * np.log(z / z0) / np.log(z_ref / z0) if np.ndim(v_ref) \
        else float(v_ref) * np.log(z / z0) / np.log(z_ref / z0)


def utci_grid(ta_c, tmrt_c, v, rh, clip_wind: bool = True):
    """Vectorised UTCI (deg C) with the same shape as the broadcast inputs.

    pythermalcomfort's UTCI regression expects wind at 10 m; v outside [0.5, 17] m/s would
    become NaN, so it is clipped first (calm street canyons -> 0.5 m/s). Inputs outside the
    other applicability limits (e.g. Tmrt - Ta > 70 K) return NaN and are counted by callers.
    """
    from pythermalcomfort.models import utci

    ta, tr, vv, hr = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (ta_c, tmrt_c, v, rh)))
    if clip_wind:
        vv = np.clip(vv, V_MIN, V_MAX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = utci(tdb=ta.ravel(), tr=tr.ravel(), v=vv.ravel(), rh=hr.ravel(),
                   limit_inputs=True, round_output=False)
    return np.asarray(res.utci, dtype=float).reshape(ta.shape)


STRESS_BANDS = [(-np.inf, -40, "extreme cold"), (-40, -27, "very strong cold"), (-27, -13, "strong cold"),
                (-13, 0, "moderate cold"), (0, 9, "slight cold"), (9, 26, "no thermal stress"),
                (26, 32, "moderate heat"), (32, 38, "strong heat"), (38, 46, "very strong heat"),
                (46, np.inf, "extreme heat")]
