"""OpenStreetMap walkable network + sidewalks + destinations via OSMnx (>= 2.0 API).

Outputs feed Block D (graph construction, Sevtsuk-style pedestrian flow estimation).
"""
from __future__ import annotations

from pathlib import Path

from . import Request

DESTINATION_TAGS = {
    "amenity": True, "shop": True, "public_transport": ["station", "stop_position", "platform"],
    "leisure": ["park", "playground"], "office": True,
}


def plan(patch, ctx) -> list[Request]:
    from ..geo import approx_wgs84_bbox

    b = approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m)
    q = f"bbox={tuple(round(v, 5) for v in b)}"
    return [
        Request("osm", f"Overpass: walk network {q}", est_mb=2.0),
        Request("osm", f"Overpass: footway=sidewalk features {q}", est_mb=0.5),
        Request("osm", f"Overpass: destinations (amenity/shop/transit/leisure/office) {q}", est_mb=1.0),
    ]


def fetch(patch, ctx) -> list[Path]:
    import osmnx as ox
    from ..geo import approx_wgs84_bbox, working_crs

    out = ctx.patch_dir(patch)
    crs = working_crs(patch.source, patch.lon, patch.lat)
    bbox = approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m)  # (W, S, E, N) = osmnx 2.x order

    G = ox.graph_from_bbox(bbox, network_type="walk", simplify=True, retain_all=False)
    G = ox.project_graph(G, to_crs=crs)
    ox.save_graphml(G, out / "walk_network.graphml")
    nodes, edges = ox.graph_to_gdfs(G)
    gpkg = out / "osm.gpkg"
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    edges.to_file(gpkg, layer="edges", driver="GPKG")

    for layer, tags in (("sidewalks", {"footway": "sidewalk"}), ("destinations", DESTINATION_TAGS)):
        try:
            f = ox.features_from_bbox(bbox, tags=tags).to_crs(crs)
            f = f[[c for c in f.columns if c in ("geometry", "amenity", "shop", "public_transport",
                                                  "leisure", "office", "footway", "name")]]
            f.to_file(gpkg, layer=layer, driver="GPKG")
        except Exception as e:  # osmnx raises when no features match
            (out / f"osm_{layer}_EMPTY.txt").write_text(str(e), encoding="utf-8")
    return [out / "walk_network.graphml", gpkg]
