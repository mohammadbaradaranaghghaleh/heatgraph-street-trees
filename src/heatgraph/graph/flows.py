"""Pedestrian flow estimation on street networks.

`gravity_flows` follows the logic of Sevtsuk (2021, JAPA): trips are generated at
origins, distributed to destinations within a walking radius in proportion to
attractiveness x distance decay (Huff-type gravity), and assigned not only to the
shortest path but spread evenly over all simple paths up to `detour_ratio` x the
shortest length (bounded by `max_paths`). Edge flows are trips per period.

`betweenness_flows` is the unweighted baseline (length-weighted edge betweenness),
used in the sensitivity analysis of the objective weighting.

Graph requirements: networkx (Multi)Graph, undirected, edge attribute `length` (m).
"""
from __future__ import annotations

import itertools
import math

import networkx as nx


def _edge_key(u, v):
    return (u, v) if str(u) <= str(v) else (v, u)


def _simple(G):
    """Collapse a MultiGraph to a simple Graph keeping the shortest parallel edge."""
    if not G.is_multigraph():
        return G.to_undirected() if G.is_directed() else G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        if not H.has_edge(u, v) or d["length"] < H[u][v]["length"]:
            H.add_edge(u, v, **d)
    return H


def gravity_flows(G, origins: dict, destinations: dict, radius_m: float = 800.0, beta: float = 0.002,
                  detour_ratio: float = 1.15, max_paths: int = 10) -> dict:
    """Return {(u, v): flow} with (u, v) in canonical order.

    origins:       node -> trip production (e.g. residents, or residents x trip rate)
    destinations:  node -> attractiveness (e.g. floor area, POI count, transit weight)
    beta:          exponential distance decay per metre (0.002 ~ halving every ~350 m)
    """
    H = _simple(G)
    flows = {_edge_key(u, v): 0.0 for u, v in H.edges()}
    dest_nodes = [d for d, a in destinations.items() if a > 0 and d in H]
    if max_paths == 1:                       # fast path: shortest paths only (one Dijkstra per origin)
        for o, production in origins.items():
            if production <= 0 or o not in H:
                continue
            dist, paths = nx.single_source_dijkstra(H, o, cutoff=radius_m, weight="length")
            reach = [(d, destinations[d] * math.exp(-beta * dist[d])) for d in dest_nodes if d in dist and d != o]
            total = sum(w for _, w in reach)
            for d, w in reach:
                p = paths[d]
                for u, v in zip(p[:-1], p[1:]):
                    flows[_edge_key(u, v)] += production * w / total
        return flows
    for o, production in origins.items():
        if production <= 0 or o not in H:
            continue
        dist = nx.single_source_dijkstra_path_length(H, o, cutoff=radius_m, weight="length")
        reach = [(d, destinations[d] * math.exp(-beta * dist[d])) for d in dest_nodes if d in dist and d != o]
        total = sum(w for _, w in reach)
        if total <= 0:
            continue
        for d, w in reach:
            trips = production * w / total
            limit = detour_ratio * dist[d] + 1e-9
            paths = []
            for path in itertools.islice(nx.shortest_simple_paths(H, o, d, weight="length"), max_paths):
                if nx.path_weight(H, path, "length") > limit:
                    break
                paths.append(path)
            share = trips / len(paths)
            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    flows[_edge_key(u, v)] += share
    return flows


def betweenness_flows(G, normalized: bool = True) -> dict:
    H = _simple(G)
    eb = nx.edge_betweenness_centrality(H, weight="length", normalized=normalized)
    return {_edge_key(u, v): val for (u, v), val in eb.items()}
