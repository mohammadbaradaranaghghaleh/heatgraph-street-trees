"""Synthetic street scenes for end-to-end tests and demos (no external data).

A scene is a small grid city: building blocks between a 3 x 3 lattice of streets, a
pedestrian network along the street centrelines, and candidate tree-planting slots on
the sidewalks. Labels for any planting are produced by the real Block C engine, so the
synthetic pipeline exercises exactly the code used on real patches.

Coordinates: pixel (row, col), 1 m pixels, row 0 = north.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import svf as svf_mod
from .engine.simulate import simulate
from .engine.surfaces import grids_from_arrays
from .graph.exposure import edge_utci_from_receptors

TOY_ENGINE_CFG = {
    "grid": {"buffer_m": 60},
    "sun": {"min_elevation_deg": 3.0},
    "shadow": {"tree_transmissivity": 0.03},
    "svf": {"n_azimuths": 16, "max_distance_m": 60},
    "wind": {"urban_reduction_factor": 0.5},
    "radiation": {},
}


@dataclass
class Scene:
    building_heights: np.ndarray
    base_canopy: np.ndarray
    nodes: dict                    # node id -> (row, col)
    edges: list                    # list of (u, v) node ids, index = segment id
    receptors: list                # per segment: (n, 2) int array of (row, col)
    slots: np.ndarray              # (K, 2) slot centres (row, col)
    crown_height: float = 8.0
    crown_radius: int = 3
    lat: float = 52.37
    lon: float = 4.90
    weather: object = None         # DataFrame indexed by UTC timestamps
    group: np.ndarray = None       # neighbourhood id per segment (equity groups)
    population: dict = field(default_factory=dict)   # group id -> residents

    @property
    def n_segments(self):
        return len(self.edges)

    @property
    def n_slots(self):
        return len(self.slots)

    def canopy_with(self, planting) -> np.ndarray:
        """Base canopy plus crowns (discs) at the planted slots."""
        c = self.base_canopy.copy()
        n = c.shape[0]
        yy, xx = np.mgrid[0:n, 0:c.shape[1]]
        for k in np.flatnonzero(np.asarray(planting) > 0):
            r, q = self.slots[k]
            disc = (yy - r) ** 2 + (xx - q) ** 2 <= self.crown_radius ** 2
            c = np.where(disc, np.maximum(c, self.crown_height), c)
        return c


def toy_scene(seed: int = 0, n: int = 80, lines=(10, 40, 70), half_street: int = 6) -> Scene:
    import pandas as pd

    rng = np.random.default_rng(seed)
    b = np.zeros((n, n), np.float32)
    for i in range(len(lines) - 1):
        for j in range(len(lines) - 1):
            r0, r1 = lines[i] + half_street + 1, lines[i + 1] - half_street
            c0, c1 = lines[j] + half_street + 1, lines[j + 1] - half_street
            b[r0:r1, c0:c1] = rng.uniform(8.0, 22.0)
    nodes = {(i, j): (lines[i], lines[j]) for i in range(len(lines)) for j in range(len(lines))}
    edges = []
    for i in range(len(lines)):
        for j in range(len(lines) - 1):
            edges.append(((i, j), (i, j + 1)))          # east-west segment
            edges.append(((j, i), (j + 1, i)))          # north-south segment
    receptors, slots = [], []
    for e, (u, v) in enumerate(edges):
        (r0, c0), (r1, c1) = nodes[u], nodes[v]
        m = max(abs(r1 - r0), abs(c1 - c0))
        t = np.arange(2, m - 1, 2) / m
        receptors.append(np.stack([np.round(r0 + t * (r1 - r0)), np.round(c0 + t * (c1 - c0))], 1).astype(int))
        side = 1 if e % 2 == 0 else -1
        for frac in (1 / 3, 2 / 3):
            rr, cc = r0 + frac * (r1 - r0), c0 + frac * (c1 - c0)
            if r0 == r1:
                rr += side * 4                          # sidewalk north/south of an E-W street
            else:
                cc += side * 4
            slots.append((int(round(rr)), int(round(cc))))
    times = pd.DatetimeIndex(["2019-07-25 09:00", "2019-07-25 11:40", "2019-07-25 14:30"], tz="UTC")
    weather = pd.DataFrame({"temperature_2m": [30.0, 35.0, 36.0], "relative_humidity_2m": [40.0, 30.0, 28.0],
                            "wind_speed_10m": [2.5, 3.0, 3.0], "shortwave_radiation": [650.0, 850.0, 720.0]},
                           index=times)
    mid = np.array([np.mean([nodes[u][1], nodes[v][1]]) for u, v in edges])
    group = (mid > n / 2).astype(int)                    # west (0) / east (1) neighbourhoods
    return Scene(b, np.zeros_like(b), nodes, edges, receptors, np.array(slots), weather=weather,
                 group=group, population={0: 1200.0, 1: 800.0})


def simulate_planting(scene: Scene, planting, cfg: dict | None = None) -> dict:
    """Run the engine for one planting; return per-segment hourly UTCI and diagnostics."""
    cfg = cfg or TOY_ENGINE_CFG
    g = grids_from_arrays(scene.building_heights, scene.canopy_with(planting))
    sky = svf_mod.compute(g.dsm, g.cdsm, g.resolution_m, cfg["svf"]["n_azimuths"], cfg["svf"]["max_distance_m"],
                          cfg["shadow"]["tree_transmissivity"])
    out = simulate(g, scene.weather, scene.lat, scene.lon, cfg, svf_cache=sky)
    seg_utci = edge_utci_from_receptors(out["utci"], scene.receptors)       # (hours, segments)
    return {"utci": seg_utci, "sim": out}


def random_plantings(scene: Scene, n: int, p: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.random((n, scene.n_slots)) < p).astype(np.int8)
    x[0] = 0                                             # always include the unplanted scene
    return x


def generate_labels(scene: Scene, plantings: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """(n_plantings, hours, segments) UTCI from the engine."""
    return np.stack([simulate_planting(scene, x, cfg)["utci"] for x in plantings])
