"""Real-patch scenes for Stage 4: network segments, receptors and candidate planting slots.

Produces an object with the same interface as `synthetic.Scene`, so SceneGraph, the
receptor engine and the optimiser work unchanged on real patches.

Segments   walk-network edges (Overture, OSM-derived) whose midpoint lies in the 500 m core
Receptors  every RECEPTOR_SPACING_M along each segment, building pixels excluded
Slots      sidewalk positions SLOT_OFFSET_M either side of street-like edges every
           SLOT_SPACING_M; rejected if the crown disc touches a building or water, if the
           centre already has canopy >= 2 m, or if closer than MIN_SLOT_GAP_M to another
           slot; at most MAX_SLOTS per patch (seeded random subset)
Hours      local LOCAL_HOURS on the heatwave days (preceding-hour radiation, see engine.toml)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine.surfaces import rasterize_polygons

RECEPTOR_SPACING_M = 5.0
SLOT_SPACING_M = 12.0
SLOT_OFFSET_M = 4.0
MIN_SLOT_GAP_M = 8.0
MAX_SLOTS = 250
CROWN_HEIGHT_M = 8.0
CROWN_RADIUS_M = 3
LOCAL_HOURS = range(8, 20)
SLOT_CLASSES = {"residential", "living_street", "pedestrian", "footway", "sidewalk", "tertiary", "secondary",
                "primary", "unclassified", "service", "cycleway", "path", "crosswalk"}


@dataclass
class RealScene:
    patch_id: str
    building_heights: np.ndarray
    base_canopy: np.ndarray
    nodes: dict
    edges: list
    receptors: list
    slots: np.ndarray
    segment_length_m: np.ndarray
    segment_class: list
    weather: object
    lat: float
    lon: float
    crown_height: float = CROWN_HEIGHT_M
    crown_radius: int = CROWN_RADIUS_M
    water_mask: np.ndarray = None
    extra: dict = field(default_factory=dict)

    @property
    def n_segments(self):
        return len(self.edges)

    @property
    def n_slots(self):
        return len(self.slots)

    def _disc(self):
        r = self.crown_radius
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return yy[(yy ** 2 + xx ** 2) <= r * r], xx[(yy ** 2 + xx ** 2) <= r * r]

    def canopy_with(self, planting) -> np.ndarray:
        """Base canopy + crown discs at planted slots (0 inside buildings), fast stamping."""
        c = self.base_canopy.copy()
        dy, dx = self._disc()
        H, W = c.shape
        for k in np.flatnonzero(np.asarray(planting) > 0):
            r, q = self.slots[k]
            rr, cc = r + dy, q + dx
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            c[rr[ok], cc[ok]] = np.maximum(c[rr[ok], cc[ok]], self.crown_height)
        return np.where(self.building_heights > 0, 0.0, c).astype(np.float32)


def _sample_line(geom, spacing):
    L = geom.length
    d = np.arange(spacing / 2, L, spacing) if L > spacing else np.array([L / 2])
    return [geom.interpolate(x) for x in d], d


def build_scene(p, day_hours: dict | None = None, seed: int = 0, max_slots: int = MAX_SLOTS) -> RealScene:
    """p: realpatch.RealPatch. day_hours: {local_day: hours}; default all heatwave days, LOCAL_HOURS."""
    import geopandas as gpd
    import pandas as pd

    from .config import DATA_DIR, load_patches

    d = DATA_DIR / "raw" / p.patch.id
    spec, g = p.spec, p.grids
    H, W = g.dsm.shape
    edges = gpd.read_file(d / "network.gpkg", layer="edges").to_crs(p.crs)
    try:
        water = gpd.read_file(d / "network.gpkg", layer="water").to_crs(p.crs)
    except Exception:
        water = None
    water_mask = (rasterize_polygons(list(water.geometry), [1.0] * len(water), spec) > 0) if water is not None \
        and len(water) else np.zeros((H, W), bool)

    def to_rc(x, y):
        c, r = spec.xy_to_colrow(x, y)
        return int(np.floor(r)), int(np.floor(c))

    core = p.core_mask
    mids = edges.geometry.interpolate(0.5, normalized=True)
    in_core = np.array([0 <= to_rc(m.x, m.y)[0] < H and 0 <= to_rc(m.x, m.y)[1] < W and core[to_rc(m.x, m.y)]
                        for m in mids])
    edges = edges[in_core & (edges["length_m"] >= 5.0)].reset_index(drop=True)

    nodes, seg_edges, receptors, lengths, classes = {}, [], [], [], []
    for e in edges.itertuples():
        pts, _ = _sample_line(e.geometry, RECEPTOR_SPACING_M)
        rc = []
        for pt in pts:
            r, c = to_rc(pt.x, pt.y)
            if 0 <= r < H and 0 <= c < W and g.dsm[r, c] <= 0 and (r, c) not in rc:
                rc.append((r, c))
        if not rc:
            continue
        (x0, y0), (x1, y1) = e.geometry.coords[0], e.geometry.coords[-1]
        nodes.setdefault(e.u, to_rc(x0, y0))
        nodes.setdefault(e.v, to_rc(x1, y1))
        seg_edges.append((e.u, e.v))
        receptors.append(np.asarray(rc, np.int64))
        lengths.append(float(e.length_m))
        classes.append(str(edges.loc[e.Index, "class"]))

    # ---- candidate planting slots
    rng = np.random.default_rng(seed)
    r_disc = CROWN_RADIUS_M
    yy, xx = np.mgrid[-r_disc:r_disc + 1, -r_disc:r_disc + 1]
    disc = (yy ** 2 + xx ** 2) <= r_disc ** 2
    blocked = (g.dsm > 0) | water_mask
    cand = []
    for e in edges.itertuples():
        if str(edges.loc[e.Index, "class"]) not in SLOT_CLASSES:
            continue
        geom = e.geometry
        for pt, dist in zip(*_sample_line(geom, SLOT_SPACING_M)):
            a = geom.interpolate(max(dist - 1.0, 0.0))
            bpt = geom.interpolate(min(dist + 1.0, geom.length))
            tx, ty = bpt.x - a.x, bpt.y - a.y
            n = np.hypot(tx, ty)
            if n == 0:
                continue
            nx_, ny_ = -ty / n, tx / n
            for s in (-1, 1):
                r, c = to_rc(pt.x + s * SLOT_OFFSET_M * nx_, pt.y + s * SLOT_OFFSET_M * ny_)
                if not (r_disc <= r < H - r_disc and r_disc <= c < W - r_disc) or not core[r, c]:
                    continue
                win = blocked[r - r_disc:r + r_disc + 1, c - r_disc:c + r_disc + 1]
                if (win & disc).any() or g.cdsm[r, c] >= 2.0:
                    continue
                cand.append((r, c))
    cand = list(dict.fromkeys(cand))
    order = rng.permutation(len(cand))
    slots = []
    for i in order:
        r, c = cand[i]
        if all((r - a) ** 2 + (c - b2) ** 2 >= MIN_SLOT_GAP_M ** 2 for a, b2 in slots):
            slots.append((r, c))
        if len(slots) >= max_slots:
            break
    slots = np.asarray(slots, np.int64).reshape(-1, 2)

    # ---- forcing: chosen local hours of the heatwave days
    meta, _ = load_patches()
    day_hours = day_hours or {dd: list(LOCAL_HOURS) for dd in meta.get("heatwave_days", [])}
    loc = p.weather.index.tz_convert("Europe/Berlin" if p.patch.source == "overture" else "Europe/Amsterdam")
    keep = np.zeros(len(loc), bool)
    for dd, hrs in day_hours.items():
        keep |= (loc.strftime("%Y-%m-%d") == dd) & np.isin(loc.hour, list(hrs))
    weather = p.weather[keep]

    return RealScene(p.patch.id, g.dsm.astype(np.float32), g.cdsm.astype(np.float32), nodes, seg_edges, receptors,
                     slots, np.asarray(lengths), classes, weather, p.patch.lat, p.patch.lon, water_mask=water_mask,
                     extra={"n_candidate_slots": len(cand), "n_edges_core": len(edges)})


# --------------------------------------------------------------------------- persistence
def save_scene(scene: RealScene, out_dir) -> None:
    """Write everything needed to rebuild the scene without the raw files."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_ids = list(scene.nodes)
    idx = {n: i for i, n in enumerate(node_ids)}
    np.savez_compressed(
        out / "scene.npz", building_heights=scene.building_heights, base_canopy=scene.base_canopy,
        water_mask=scene.water_mask, receptors=np.vstack(scene.receptors),
        seg_of_receptor=np.concatenate([np.full(len(r), e) for e, r in enumerate(scene.receptors)]),
        node_rc=np.array([scene.nodes[n] for n in node_ids], np.int64),
        edges=np.array([[idx[u], idx[v]] for u, v in scene.edges], np.int64).reshape(-1, 2),
        slots=scene.slots, segment_length_m=scene.segment_length_m)
    scene.weather.to_csv(out / "weather.csv", index_label="time_utc")
    meta = {"patch_id": scene.patch_id, "lat": scene.lat, "lon": scene.lon, "crown_height": scene.crown_height,
            "crown_radius": scene.crown_radius, "segment_class": scene.segment_class, **scene.extra}
    (out / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_scene(in_dir) -> RealScene:
    import json
    from pathlib import Path

    import pandas as pd

    d = Path(in_dir)
    z = np.load(d / "scene.npz")
    meta = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    w = pd.read_csv(d / "weather.csv", parse_dates=["time_utc"])
    t = pd.DatetimeIndex(w.pop("time_utc"))
    w.index = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    seg = z["seg_of_receptor"]
    rec = z["receptors"]
    receptors = [rec[seg == e] for e in range(len(z["edges"]))]
    nodes = {i: tuple(rc) for i, rc in enumerate(z["node_rc"])}
    edges = [tuple(e) for e in z["edges"]]
    return RealScene(meta["patch_id"], z["building_heights"], z["base_canopy"], nodes, edges, receptors, z["slots"],
                     z["segment_length_m"], meta["segment_class"], w, meta["lat"], meta["lon"],
                     meta["crown_height"], meta["crown_radius"], z["water_mask"],
                     {k: v for k, v in meta.items() if k.startswith("n_")})
