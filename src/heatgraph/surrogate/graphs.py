"""Heterogeneous graph construction for the surrogate (Block E).

Node types
  segment  pedestrian street segment (prediction target: hourly UTCI)
  slot     candidate planting slot (only planted slots are wired to segments)
Edge types
  (segment, adj, segment)    segments sharing an intersection (both directions)
  (slot, shades, segment)    planted slot within `link_radius_m` of a segment's nearest
                             receptor, with physics-informed, hour-dependent edge features:
                             distance, alignment with the shadow direction, and ray-height
                             ratio dist*tan(e)/crown_height (< 1: the crown can shade that point)
Segment features = static morphology + hourly building-shadow fraction + hourly forcing.
Building-only SVF and shadow come from the receptor engine (exactly the engine's values at
the receptors), so they do not depend on the planting; the surrogate learns what trees add.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..engine import sun
from ..engine.receptor_engine import ReceptorEngine, segment_mean

SEG_STATIC = ["length_100m", "sin2theta", "cos2theta", "bldg_height_20m", "svf_b"]
SEG_HOURLY = ["bldg_sunlit", "elev_90", "sin_az", "cos_az", "ta_40", "rh_100", "dni_1000", "dhi_500", "wind_10"]
SLOT_FEATURES = ["planted", "crown_h_10", "crown_r_5"]
EDGE_FEATURES = ["dist_norm", "shadow_align", "shadow_cross", "ray_ratio", "crown_shade_frac", "overhead_frac"]
# crown_shade_frac (v2): share of the segment's receptors inside the crown's geometric ground
#   shadow this hour (a "stadium": discs of crown radius centred between trunk-top and crown-top
#   shadow points) AND not already in building shadow -> what a tree adds, by construction.
# overhead_frac (v2): share of the segment's receptors under the crown (sky-view reduction).
# Both describe the mechanism, not the city, which is why they should transfer across cities.


def _local_positive_mean(dsm, rc, half: int = 20):
    """Per receptor: (sum, count) of positive heights in a (2*half+1)^2 window (zero padding),
    via a summed-area table; equals the pooled mean of stacked windows."""
    pos = (dsm > 0).astype(np.float64)
    val = np.where(dsm > 0, dsm, 0.0).astype(np.float64)

    def sat(a):
        p = np.pad(a, ((half + 1, half), (half + 1, half)))
        return p.cumsum(0).cumsum(1)

    S, C = sat(val), sat(pos)
    r, c = rc[:, 0] + half + 1, rc[:, 1] + half + 1        # window rows r-half..r+half in padded coords

    def box(T):
        r0, r1, c0, c1 = r - half - 1, r + half, c - half - 1, c + half
        return T[r1, c1] - T[r0, c1] - T[r1, c0] + T[r0, c0]

    return box(S), box(C)


class SceneGraph:
    """Precomputes everything that does not depend on the planting, then builds
    a HeteroData per (planting, hour) in about a millisecond."""

    def __init__(self, scene, baseline_utci=None, link_radius_m: float = 25.0, cfg: dict | None = None,
                 engine: ReceptorEngine | None = None, device: str = "cpu", physics_edges: bool = True):
        """baseline_utci: (hours, segments) engine UTCI of the UNPLANTED scene. When given,
        the surrogate predicts only the tree-induced change on top of it (residual learning)."""
        import pandas as pd
        from ..synthetic import TOY_ENGINE_CFG

        cfg = cfg or TOY_ENGINE_CFG
        self.scene, self.link_radius = scene, link_radius_m
        self.baseline = None if baseline_utci is None else np.asarray(baseline_utci, np.float32)
        rc = np.vstack(scene.receptors).astype(np.int64)
        self.seg_of_receptor = np.concatenate([np.full(len(r), e) for e, r in enumerate(scene.receptors)])
        E = scene.n_segments
        if engine is None:
            engine = ReceptorEngine(scene.building_heights, rc, scene.weather, scene.lat, scene.lon, cfg, device=device)
        self.engine = engine
        w = scene.weather
        self.hours = len(w)
        self.elev, self.az = engine.elev, engine.az
        dni, dhi = engine.dni, engine.dhi

        # ---- static segment features
        nodes = scene.nodes
        s_sum, s_cnt = _local_positive_mean(np.asarray(scene.building_heights, np.float64), rc)
        seg_sum = np.bincount(self.seg_of_receptor, s_sum, E)
        seg_cnt = np.bincount(self.seg_of_receptor, s_cnt, E)
        hmean = np.where(seg_cnt > 0, seg_sum / np.maximum(seg_cnt, 1), 0.0)
        svf_b = segment_mean(engine.svf_b.cpu().numpy()[None], self.seg_of_receptor, E)[0]
        lengths = getattr(scene, "segment_length_m", None)
        seg_static = []
        for e, (u, v) in enumerate(scene.edges):
            a, b = np.array(nodes[u], float), np.array(nodes[v], float)
            d = b - a
            theta = math.atan2(d[1], -d[0])                       # bearing from north (row up = north)
            L = float(lengths[e]) if lengths is not None else float(np.linalg.norm(d))
            seg_static.append([L / 100.0, math.sin(2 * theta), math.cos(2 * theta), hmean[e] / 20.0, svf_b[e]])
        self.seg_static = np.asarray(seg_static, np.float32)

        bld_sun = segment_mean((~engine.bshadow).float().cpu().numpy(), self.seg_of_receptor, E)   # (hours, E)
        self.seg_hourly = np.stack([np.column_stack([
            bld_sun[h],
            np.full(E, self.elev[h] / 90.0), np.full(E, math.sin(math.radians(self.az[h]))),
            np.full(E, math.cos(math.radians(self.az[h]))),
            np.full(E, w["temperature_2m"].iloc[h] / 40.0),
            np.full(E, w["relative_humidity_2m"].iloc[h] / 100.0),
            np.full(E, dni[h] / 1000.0), np.full(E, dhi[h] / 500.0),
            np.full(E, w["wind_speed_10m"].iloc[h] / 10.0)]) for h in range(self.hours)]).astype(np.float32)

        # ---- segment adjacency (shared intersection), via a node -> segments index
        by_node = {}
        for i, (u, v) in enumerate(scene.edges):
            by_node.setdefault(u, set()).add(i)
            by_node.setdefault(v, set()).add(i)
        pairs = {(i, j) for segs in by_node.values() for i in segs for j in segs if i != j}
        src, dst = zip(*sorted(pairs)) if pairs else ((), ())
        self.adj = torch.tensor([list(src), list(dst)], dtype=torch.long)

        # ---- slot -> segment links: nearest receptor of each segment within the radius
        slots = np.asarray(scene.slots, float)
        rcf = rc.astype(float)
        link_slot, link_seg, link_dist, link_vec = [], [], [], []
        for k, p in enumerate(slots):
            d = np.hypot(rcf[:, 0] - p[0], rcf[:, 1] - p[1])
            near = d <= link_radius_m
            if not near.any():
                continue
            segs = self.seg_of_receptor[near]
            idx = np.flatnonzero(near)
            order = np.lexsort((d[idx], segs))                     # per segment, nearest receptor first
            segs_o, idx_o = segs[order], idx[order]
            first = np.r_[True, segs_o[1:] != segs_o[:-1]]
            for e, j in zip(segs_o[first], idx_o[first]):
                vec = rcf[j] - p
                n = np.linalg.norm(vec)
                link_slot.append(k); link_seg.append(e); link_dist.append(d[j])
                link_vec.append(vec / n if n > 0 else np.zeros(2))
        self.link_slot = np.asarray(link_slot, np.int64)
        self.link_seg = np.asarray(link_seg, np.int64)
        dist = np.asarray(link_dist, float)
        vec = np.asarray(link_vec, float).reshape(-1, 2)
        self.link_attr = np.zeros((self.hours, len(dist), len(EDGE_FEATURES)), np.float32)

        # (link, receptor) pairs for the crown-shadow / overhead features
        rec_of_seg = [np.flatnonzero(self.seg_of_receptor == e) for e in range(E)]
        pair_link = np.concatenate([np.full(len(rec_of_seg[e]), j) for j, e in enumerate(self.link_seg)]) \
            if len(self.link_seg) else np.zeros(0, np.int64)
        pair_rec = np.concatenate([rec_of_seg[e] for e in self.link_seg]) if len(self.link_seg) else np.zeros(0, np.int64)
        n_rec_link = np.bincount(pair_link, minlength=len(self.link_seg)).astype(float) if len(pair_link) else np.ones(0)
        pv = rcf[pair_rec] - slots[self.link_slot[pair_link]] if len(pair_link) else np.zeros((0, 2))   # slot -> receptor
        R = float(scene.crown_radius)
        trunk = cfg["shadow"].get("trunk_zone_fraction", 0.25) * scene.crown_height
        bsh = engine.bshadow.cpu().numpy()                         # (hours, N) building shadow at receptors
        min_el = cfg["sun"]["min_elevation_deg"]

        def per_link(x):
            return np.bincount(pair_link, weights=x.astype(float), minlength=len(self.link_seg)) / np.maximum(n_rec_link, 1)

        overhead = per_link(np.hypot(pv[:, 0], pv[:, 1]) <= R) if len(pair_link) else np.zeros(0)
        for h in range(self.hours):
            dr, dc = sun.sun_vector_grid(self.az[h])
            away = -np.array([dr, dc])                             # shadows fall away from the sun
            tan_e = math.tan(math.radians(max(self.elev[h], 0.5)))
            cols = [dist / link_radius_m, vec @ away, vec[:, 0] * away[1] - vec[:, 1] * away[0],
                    np.minimum(dist * tan_e / scene.crown_height, 5.0) / 5.0]
            if physics_edges and len(pair_link) and self.elev[h] > min_el:
                s = pv @ away
                s_c = np.clip(s, trunk / tan_e, scene.crown_height / tan_e)
                perp = pv - s_c[:, None] * away[None]
                in_shadow = np.hypot(perp[:, 0], perp[:, 1]) <= R
                shade = per_link(in_shadow & ~bsh[h, pair_rec])
            else:
                shade = np.zeros(len(dist))
            cols += [shade, overhead if physics_edges else np.zeros(len(dist))]
            self.link_attr[h] = np.column_stack(cols)

    def build(self, planting, hour: int, y=None) -> HeteroData:
        x = np.asarray(planting) > 0
        data = HeteroData()
        data["segment"].x = torch.from_numpy(np.hstack([self.seg_static, self.seg_hourly[hour]]))
        sc = self.scene
        data["slot"].x = torch.from_numpy(np.column_stack([
            x.astype(np.float32), np.full(sc.n_slots, sc.crown_height / 10.0, np.float32),
            np.full(sc.n_slots, sc.crown_radius / 5.0, np.float32)]))
        keep = x[self.link_slot]
        data["slot", "shades", "segment"].edge_index = torch.from_numpy(
            np.vstack([self.link_slot[keep], self.link_seg[keep]]).astype(np.int64))
        data["slot", "shades", "segment"].edge_attr = torch.from_numpy(self.link_attr[hour][keep])
        data["segment", "adj", "segment"].edge_index = self.adj
        data["segment"].hour = torch.full((sc.n_segments,), hour, dtype=torch.long)   # for per-hour models
        if self.baseline is not None:
            data["segment"].base = torch.from_numpy(self.baseline[hour])
        if y is not None:
            data["segment"].y = torch.as_tensor(np.asarray(y, np.float32))
        return data

    def build_batch(self, plantings, hours=None) -> HeteroData:
        """One disjoint-union graph for every (planting, hour) pair, built with array offsets
        (orders of magnitude faster than Batch.from_data_list for optimisation). Node order:
        planting-major, then hour, then segment -> predictions reshape to (n, len(hours), E)."""
        X = np.atleast_2d(np.asarray(plantings) > 0)
        hours = list(range(self.hours)) if hours is None else list(hours)
        n, H, sc = len(X), len(hours), self.scene
        E, K = sc.n_segments, sc.n_slots
        G = n * H
        seg_x = np.concatenate([np.hstack([self.seg_static, self.seg_hourly[h]]) for h in hours])   # (H*E, F)
        data = HeteroData()
        data["segment"].x = torch.from_numpy(np.tile(seg_x, (n, 1)))
        slot_static = np.column_stack([np.full(K, sc.crown_height / 10.0), np.full(K, sc.crown_radius / 5.0)])
        planted = np.repeat(X, H, axis=0).reshape(-1).astype(np.float32)                           # (G*K,)
        data["slot"].x = torch.from_numpy(np.column_stack([planted, np.tile(slot_static, (G, 1))]).astype(np.float32))
        src, dst, attr = [], [], []
        for gi in range(G):
            i, h = divmod(gi, H)
            keep = X[i][self.link_slot]
            src.append(self.link_slot[keep] + gi * K)
            dst.append(self.link_seg[keep] + gi * E)
            attr.append(self.link_attr[hours[h]][keep])
        data["slot", "shades", "segment"].edge_index = torch.from_numpy(
            np.vstack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64))
        data["slot", "shades", "segment"].edge_attr = torch.from_numpy(np.concatenate(attr).astype(np.float32))
        A = self.adj.numpy()
        off = (np.arange(G) * E)[:, None, None]
        data["segment", "adj", "segment"].edge_index = torch.from_numpy((A[None] + off).transpose(1, 0, 2).reshape(2, -1))
        data["segment"].hour = torch.from_numpy(np.tile(np.repeat(np.asarray(hours), E), n).astype(np.int64))
        if self.baseline is not None:
            data["segment"].base = torch.from_numpy(np.tile(np.concatenate([self.baseline[h] for h in hours]), n))
        return data

    def dataset(self, plantings, labels=None) -> list:
        """One HeteroData per (planting, hour); labels shaped (n, hours, segments)."""
        out = []
        for i, x in enumerate(plantings):
            for h in range(self.hours):
                out.append(self.build(x, h, None if labels is None else labels[i, h]))
        return out


class LazyGraphDataset(torch.utils.data.Dataset):
    """(scene_graph, planting, hour, label) items built on demand: keeps RAM flat when a real
    dataset has tens of thousands of (planting, hour) graphs."""

    def __init__(self, items):
        self.items = items        # list of (SceneGraph, planting (K,), hour, y (E,) or None)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if i >= len(self.items):
            raise IndexError(i)
        sg, x, h, y = self.items[i]
        return sg.build(x, h, y)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
