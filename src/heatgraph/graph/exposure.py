"""Person-weighted heat dose on pedestrian edges (the HeatGraph objective).

dose_e = sum_h  flow_{e,h} * (length_e / walking_speed) * max(UTCI_{e,h} - threshold, 0) * V_e

units: degC-hours of exceedance per person summed over people (degC * person * h).
threshold defaults to 26 degC, the lower bound of the UTCI "moderate heat stress" band.
"""
from __future__ import annotations

import numpy as np

WALK_SPEED_MS = 1.3


def edge_heat_dose(utci_hourly, flow_hourly, length_m, vulnerability=1.0, threshold_c: float = 26.0,
                   walking_speed_ms: float = WALK_SPEED_MS):
    """Arrays broadcast as (hours, edges) for utci/flow and (edges,) for length/vulnerability.

    Returns per-edge dose (edges,). NaN UTCI (outside model limits) raises, so that it
    cannot silently count as zero exposure.
    """
    u = np.asarray(utci_hourly, dtype=float)
    if np.isnan(u).any():
        raise ValueError("UTCI contains NaN; resolve out-of-range inputs before computing dose")
    f = np.asarray(flow_hourly, dtype=float)
    hours_on_edge = np.asarray(length_m, dtype=float) / walking_speed_ms / 3600.0
    excess = np.maximum(u - threshold_c, 0.0)
    return (f * excess).sum(axis=0) * hours_on_edge * np.asarray(vulnerability, dtype=float)


def edge_utci_from_receptors(utci_grid_hourly, receptor_rc: list[np.ndarray], stat: str = "mean"):
    """Aggregate gridded UTCI (hours, rows, cols) to edges from receptor pixel indices.

    receptor_rc[e] is an (n_e, 2) int array of (row, col) receptor pixels of edge e.
    Returns (hours, edges).
    """
    g = np.asarray(utci_grid_hourly)
    agg = {"mean": np.nanmean, "p90": lambda a, axis: np.nanpercentile(a, 90, axis=axis),
           "max": np.nanmax}[stat]
    cols = [agg(g[:, rc[:, 0], rc[:, 1]], axis=1) for rc in receptor_rc]
    return np.stack(cols, axis=1)


def sample_receptors_along_line(coords_xy: np.ndarray, spacing_m: float, transform_inv) -> np.ndarray:
    """Points every `spacing_m` along a polyline (x, y), converted to (row, col) pixels.

    transform_inv maps (x, y) -> (col, row) floats (e.g. ~affine_transform for rasterio).
    """
    xy = np.asarray(coords_xy, dtype=float)
    seg = np.diff(xy, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    d = np.arange(spacing_m / 2, total, spacing_m) if total > spacing_m else np.array([total / 2])
    x = np.interp(d, cum, xy[:, 0])
    y = np.interp(d, cum, xy[:, 1])
    cr = np.array([transform_inv(xi, yi) for xi, yi in zip(x, y)])
    return np.floor(cr[:, ::-1]).astype(int)   # (row, col)
