"""Patch geometry helpers.

`approx_wgs84_bbox` is stdlib-only (used for dry-run planning).
`projected_bbox` needs pyproj and is used for real requests.
"""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8
RD_NEW = "EPSG:28992"   # Dutch national grid; native CRS of 3DBAG


def approx_wgs84_bbox(lon: float, lat: float, half_size_m: float) -> tuple[float, float, float, float]:
    """Square bbox around a centre on a spherical Earth (W, S, E, N).

    Error is well under 1 m for 1 km boxes at mid latitudes, enough for request planning.
    """
    dlat = math.degrees(half_size_m / EARTH_RADIUS_M)
    dlon = math.degrees(half_size_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def projected_bbox(lon: float, lat: float, half_size_m: float, crs: str) -> tuple[float, float, float, float]:
    """Exact square bbox in a projected CRS (minx, miny, maxx, maxy)."""
    from pyproj import Transformer  # lazy: heavy dependency

    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(lon, lat)
    return (x - half_size_m, y - half_size_m, x + half_size_m, y + half_size_m)


def working_crs(source: str, lon: float, lat: float) -> str:
    """RD New for Dutch 3DBAG patches, local UTM elsewhere."""
    return RD_NEW if source == "3dbag" else utm_epsg(lon, lat)
