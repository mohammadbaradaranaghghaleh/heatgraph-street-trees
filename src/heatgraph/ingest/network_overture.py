"""Walkable street network + destinations from Overture Maps (transportation `segment`
and `place` themes), as the default replacement for live Overpass queries.

Why: the public Overpass endpoint answered our scripted queries with "at capacity" and
states that automated or high-volume use should run a local Overpass instance. Overture's
transportation layer is derived from OpenStreetMap (cite both in the data statement), ships
as GeoParquet on S3 and gives each segment an ordered list of `connectors` (node id +
linear position `at`), from which a routable graph is built directly.

Outputs in data/raw/<patch>/:
  network.gpkg          layers `edges` (u, v, class, length_m, geometry), `nodes`, `destinations`
  walk_network.graphml  networkx MultiGraph (node x/y in the working CRS, edge length)
"""
from __future__ import annotations

from pathlib import Path

from . import Request

# Overture road classes a pedestrian can use. Motorways/trunks are excluded; steps are kept
# but flagged (they matter for accessibility analyses).
WALK_CLASSES = {"footway", "sidewalk", "crosswalk", "pedestrian", "path", "steps", "living_street",
                "residential", "service", "unclassified", "tertiary", "secondary", "primary", "cycleway",
                "track", "bridleway", "unknown", "alley", "driveway", "parking_aisle"}
MIN_PLACE_CONFIDENCE = 0.5


def plan(patch, ctx) -> list[Request]:
    from ..geo import approx_wgs84_bbox

    b = tuple(round(v, 5) for v in approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m))
    return [Request("overture", f"overturemaps segment bbox={b}", est_mb=2.0, note="walk network (OSM-derived)"),
            Request("overture", f"overturemaps place bbox={b}", est_mb=2.0, note="destinations")]


def _gdf(kind, bbox):
    try:
        from overturemaps import core
        return core.geodataframe(kind, bbox=bbox)
    except (ImportError, AttributeError):
        from overturemaps import geodataframe
        return geodataframe(kind, bbox=bbox)


def segments_to_edges(seg, crs):
    """Split each segment at its connectors -> list of dict(u, v, class, geometry) in `crs`."""
    import geopandas as gpd
    from shapely.ops import substring

    seg = seg[(seg["subtype"] == "road") & seg["class"].isin(WALK_CLASSES)].to_crs(crs)
    rows = []
    for _, s in seg.iterrows():
        con = s["connectors"]
        if con is None or len(con) < 2:
            continue
        con = sorted(((c["connector_id"], float(c["at"])) for c in con), key=lambda t: t[1])
        for (u, a), (v, b) in zip(con[:-1], con[1:]):
            if b - a <= 1e-9 or u == v:
                continue
            g = substring(s.geometry, a, b, normalized=True)
            if g.is_empty or g.length == 0:
                continue
            rows.append({"u": u, "v": v, "class": s["class"], "segment_id": s["id"], "length_m": g.length,
                         "geometry": g})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def edges_to_graph(edges):
    import networkx as nx

    G = nx.MultiGraph()
    for u, v, cls, sid, length, geom in zip(edges["u"], edges["v"], edges["class"], edges["segment_id"],
                                            edges["length_m"], edges.geometry):
        x0, y0 = geom.coords[0]
        x1, y1 = geom.coords[-1]
        G.add_node(u, x=x0, y=y0)
        G.add_node(v, x=x1, y=y1)
        G.add_edge(u, v, length=float(length), cls=str(cls), segment_id=str(sid))
    return G


def fetch(patch, ctx) -> list[Path]:
    import geopandas as gpd
    import networkx as nx
    from ..geo import approx_wgs84_bbox, working_crs

    out = ctx.patch_dir(patch)
    crs = working_crs(patch.source, patch.lon, patch.lat)
    bbox = approx_wgs84_bbox(patch.lon, patch.lat, ctx.half_size_m)

    edges = segments_to_edges(_gdf("segment", bbox).set_crs("EPSG:4326", allow_override=True), crs)
    if edges.empty:
        raise ValueError("no walkable Overture segments in the window")
    G = edges_to_graph(edges)
    nodes = gpd.GeoDataFrame({"id": list(G.nodes)}, geometry=gpd.points_from_xy(
        [G.nodes[n]["x"] for n in G.nodes], [G.nodes[n]["y"] for n in G.nodes]), crs=crs)

    places = _gdf("place", bbox).set_crs("EPSG:4326", allow_override=True)
    if "confidence" in places:
        places = places[places["confidence"].fillna(0) >= MIN_PLACE_CONFIDENCE]
    keep = [c for c in ("id", "basic_category", "confidence", "geometry") if c in places]
    places = places[keep].to_crs(crs)

    # water polygons: planting-slot masking and patch QA (water fraction)
    water = _gdf("water", bbox).set_crs("EPSG:4326", allow_override=True)
    water = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    water = water[[c for c in ("id", "subtype", "class", "geometry") if c in water]].to_crs(crs)

    gpkg = out / "network.gpkg"
    edges.to_file(gpkg, layer="edges", driver="GPKG")
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    places.to_file(gpkg, layer="destinations", driver="GPKG")
    if len(water):
        water.to_file(gpkg, layer="water", driver="GPKG")
    gml = out / "walk_network.graphml"
    nx.write_graphml(G, gml)
    return [gpkg, gml]
