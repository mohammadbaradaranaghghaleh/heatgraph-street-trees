"""Mean radiant temperature from six-directional short- and long-wave fluxes.

Standing person, angular factors 0.06 (up, down) and 0.22 (four lateral directions)
(Thorsson et al., 2007; SOLWEIG, Lindberg et al., 2008):

    Sstr = a_k * [ sum_i F_i K_i + f_p * I * sunlit ] + a_l * sum_i F_i L_i
    Tmrt = (Sstr / (a_l * sigma)) ** 0.25 - 273.15

with the direct beam handled through the projected-area factor of a standing person
f_p = 0.308 cos(e (1 - e^2/48402)), e in degrees (Fanger; as used in RayMan/UTCI work).

Simplifications versus SOLWEIG (documented for the methods section, calibrated in Block I):
  * walls reflect/emit isotropically at a single temperature Tw = Ta + k_w * G/1000;
  * ground temperature Tg = Ta + k_g * (I sin e * sunlit + 0.1 D)/1000 (sunlit warming);
  * lateral fluxes are half sky/walls and half ground (no per-wall-aspect split yet).
"""
from __future__ import annotations

import numpy as np

SIGMA = 5.670374419e-8
F_UPDOWN, F_SIDE = 0.06, 0.22

DEFAULTS = {
    "albedo_ground": 0.15, "albedo_wall": 0.20,
    "emissivity_ground": 0.95, "emissivity_wall": 0.90,
    "absorption_shortwave": 0.70, "absorption_longwave": 0.97,
    "ground_warming_k_per_kwm2": 15.0,   # [tune] sunlit asphalt ~ +12 K at 800 W/m2
    "wall_warming_k_per_kwm2": 4.0,      # [tune]
}


def projected_area_factor(elevation_deg):
    e = np.clip(np.asarray(elevation_deg, dtype=float), 0.0, 90.0)
    return 0.308 * np.cos(np.radians(e * (1.0 - e ** 2 / 48402.0)))


def vapour_pressure_hpa(ta_c, rh_pct):
    """Actual vapour pressure (hPa), Magnus formula over water."""
    ta = np.asarray(ta_c, dtype=float)
    return np.asarray(rh_pct, dtype=float) / 100.0 * 6.112 * np.exp(17.62 * ta / (243.12 + ta))


def sky_emissivity_prata(ta_c, rh_pct):
    """Clear-sky emissivity, Prata (1996): 1 - (1 + w) exp(-sqrt(1.2 + 3 w)), w = 46.5 e_a / T."""
    w = 46.5 * vapour_pressure_hpa(ta_c, rh_pct) / (np.asarray(ta_c, dtype=float) + 273.15)
    return 1.0 - (1.0 + w) * np.exp(-np.sqrt(1.2 + 3.0 * w))


def tmrt(sunlit, svf, ta_c: float, rh_pct: float, dni: float, dhi: float, elevation_deg: float,
         params: dict | None = None):
    """Tmrt (deg C) for every pixel. `sunlit` in [0,1] (0 = building shade), `svf` in [0,1]."""
    p = {**DEFAULTS, **(params or {})}
    sunlit = np.asarray(sunlit, dtype=np.float64)
    svf = np.clip(np.asarray(svf, dtype=np.float64), 0.0, 1.0)
    e = max(float(elevation_deg), 0.0)
    sin_e = np.sin(np.radians(e))
    I = float(dni) if e > 0 else 0.0
    D = float(dhi)
    G = I * sin_e + D
    ta_k = ta_c + 273.15

    # --- shortwave
    beam_h = I * sin_e * sunlit                            # direct on horizontal ground
    wall_refl = p["albedo_wall"] * (1.0 - svf) * 0.5 * G   # walls: roughly half sunlit
    k_down = D * svf + wall_refl
    k_up = p["albedo_ground"] * (beam_h + D * svf)
    k_side = 0.5 * (D * svf + wall_refl) + 0.5 * k_up
    k_dir = projected_area_factor(e) * I * sunlit

    # --- longwave
    l_sky = sky_emissivity_prata(ta_c, rh_pct) * SIGMA * ta_k ** 4
    tw_k = ta_k + p["wall_warming_k_per_kwm2"] * G / 1000.0
    tg_k = ta_k + p["ground_warming_k_per_kwm2"] * (beam_h + 0.1 * D) / 1000.0
    l_wall = p["emissivity_wall"] * SIGMA * tw_k ** 4
    l_down = svf * l_sky + (1.0 - svf) * l_wall
    l_up = p["emissivity_ground"] * SIGMA * tg_k ** 4 + (1.0 - p["emissivity_ground"]) * l_down
    l_side = 0.5 * l_down + 0.5 * l_up

    ak, al = p["absorption_shortwave"], p["absorption_longwave"]
    s_str = (ak * (F_UPDOWN * (k_down + k_up) + 4 * F_SIDE * k_side + k_dir)
             + al * (F_UPDOWN * (l_down + l_up) + 4 * F_SIDE * l_side))
    return (s_str / (al * SIGMA)) ** 0.25 - 273.15
