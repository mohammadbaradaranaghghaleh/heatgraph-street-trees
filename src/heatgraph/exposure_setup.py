"""Exposure set-up for optimisation on real patches: pedestrian flows, districts, population.

flows        gravity model on the full 900 m walk network (Sevtsuk-style, shortest paths):
             origins = every network node with equal production (no residential data yet),
             destinations = Overture places snapped to the nearest node (count = attractiveness),
             walking radius 800 m, exponential decay beta = 0.002 / m. Constant over hours.
districts    the 500 m core split into 2 x 2 quadrants; each segment belongs to the quadrant
             of its midpoint (equity groups).
population   PROXY: building floor area per district from 3DBAG heights (footprint m2 x
             max(1, height / 3 m)). This is a stand-in for residents until CBS/WorldPop
             population is ingested and must be labelled as such in the paper.
vulnerability 1 for every segment (no age-structure data ingested yet).
"""
from __future__ import annotations

import numpy as np

WALK_RADIUS_M, BETA = 800.0, 0.002


def segment_flows(patch_id: str, scene) -> np.ndarray:
    """Per-segment flow for a scene built from raw data (original node ids)."""
    import geopandas as gpd
    import networkx as nx
    from scipy.spatial import cKDTree

    from .config import DATA_DIR
    from .graph.flows import gravity_flows

    d = DATA_DIR / "raw" / patch_id
    G = nx.read_graphml(d / "walk_network.graphml")
    for _, _, data in G.edges(data=True):
        data["length"] = float(data["length"])
    ids = list(G.nodes)
    xy = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in ids])
    dest = gpd.read_file(d / "network.gpkg", layer="destinations")
    _, near = cKDTree(xy).query(np.column_stack([dest.geometry.x, dest.geometry.y]))
    attract = {}
    for j in near:
        attract[ids[j]] = attract.get(ids[j], 0.0) + 1.0
    f = gravity_flows(G, {n: 1.0 for n in ids}, attract, radius_m=WALK_RADIUS_M, beta=BETA, max_paths=1)
    key = lambda u, v: (u, v) if str(u) <= str(v) else (v, u)
    return np.array([f.get(key(str(u), str(v)), 0.0) for u, v in scene.edges], float)


def districts(scene, core_mask) -> np.ndarray:
    rows, cols = np.nonzero(core_mask)
    r_mid, c_mid = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    out = []
    for rc in scene.receptors:
        r, c = rc.mean(0)
        out.append(int(r >= r_mid) * 2 + int(c >= c_mid))
    return np.asarray(out)


def floor_area_population(scene, core_mask) -> dict:
    rows, cols = np.nonzero(core_mask)
    r_mid, c_mid = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    h = scene.building_heights
    floors = np.where(h > 0, np.maximum(1, np.round(h / 3.0)), 0)
    rr, cc = np.indices(h.shape)
    q = (rr >= r_mid).astype(int) * 2 + (cc >= c_mid).astype(int)
    pop = {k: float(floors[core_mask & (q == k)].sum()) for k in range(4)}
    return {k: max(v, 1.0) for k, v in pop.items()}
