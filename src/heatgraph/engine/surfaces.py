"""Block B: surface grids on the 1 m working grid.

`grids_from_arrays` is the numpy core (used by tests and by `build_grids`).
`build_grids` rasterises building footprints with numpy + shapely only (pixel-centre rule);
geopandas is needed only to read a vector FILE, rasterio only to resample a canopy RASTER.
Heights are metres above local ground (flat-terrain assumption inside a 900 m window;
a DEM can be added to `dsm` later without changing downstream code).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grids:
    dsm: np.ndarray            # ground (0) + building heights
    cdsm: np.ndarray           # canopy top height, 0 where no tree, 0 under buildings
    tdsm: np.ndarray           # trunk-zone top = trunk_zone_fraction * cdsm
    building_mask: np.ndarray  # bool
    wall_height: np.ndarray    # wall height on building edge pixels, else 0
    wall_aspect: np.ndarray    # outward wall normal, deg clockwise from north (NaN off-wall)
    resolution_m: float
    transform: object = None   # affine transform (when built from rasters)
    crs: str | None = None


def grids_from_arrays(building_heights: np.ndarray, canopy_heights: np.ndarray | None = None,
                      resolution_m: float = 1.0, trunk_zone_fraction: float = 0.25,
                      min_building_height_m: float = 2.0) -> Grids:
    dsm = np.nan_to_num(np.asarray(building_heights, dtype=np.float32), nan=0.0)
    dsm = np.where(dsm >= min_building_height_m, dsm, 0.0).astype(np.float32)
    mask = dsm > 0
    if canopy_heights is None:
        cdsm = np.zeros_like(dsm)
    else:
        cdsm = np.nan_to_num(np.asarray(canopy_heights, dtype=np.float32), nan=0.0)
        cdsm = np.where(mask, 0.0, np.maximum(cdsm, 0.0)).astype(np.float32)
    tdsm = (trunk_zone_fraction * cdsm).astype(np.float32)

    # wall height = drop to the lowest 4-neighbour, on building pixels only
    p = np.pad(dsm, 1, mode="edge")
    neigh_min = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    wall_h = np.where(mask, np.maximum(dsm - neigh_min, 0.0), 0.0).astype(np.float32)

    # outward normal = minus gradient of the building mask (north = -row, east = +col)
    g_row, g_col = np.gradient(mask.astype(np.float32))
    north, east = g_row, -g_col
    aspect = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    aspect = np.where(wall_h > 0, aspect, np.nan).astype(np.float32)
    return Grids(dsm, cdsm, tdsm, mask, wall_h, aspect, float(resolution_m))


@dataclass(frozen=True)
class GridSpec:
    """North-up grid: pixel (r, c) has its centre at
    x = minx + (c + 0.5) * res, y = maxy - (r + 0.5) * res."""
    minx: float
    maxy: float
    res: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, bounds, res: float) -> "GridSpec":
        minx, miny, maxx, maxy = bounds
        w, h = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
        if w <= 0 or h <= 0:
            raise ValueError(f"empty grid for bounds {bounds}")
        return cls(float(minx), float(maxy), float(res), w, h)

    @property
    def bounds(self):
        return (self.minx, self.maxy - self.height * self.res, self.minx + self.width * self.res, self.maxy)

    def centres(self, r0=0, r1=None, c0=0, c1=None):
        r1 = self.height if r1 is None else r1
        c1 = self.width if c1 is None else c1
        xs = self.minx + (np.arange(c0, c1) + 0.5) * self.res
        ys = self.maxy - (np.arange(r0, r1) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def xy_to_colrow(self, x, y):
        """Continuous (col, row); floor() gives the pixel. Usable as exposure transform_inv."""
        return (np.asarray(x) - self.minx) / self.res, (self.maxy - np.asarray(y)) / self.res


def rasterize_polygons(geoms, values, spec: GridSpec, merge: str = "max", fill: float = 0.0) -> np.ndarray:
    """Burn polygon values into a grid by the pixel-centre rule (centre strictly inside the
    polygon, holes respected). Overlaps: 'max' (default, the tallest part wins) or 'last'.
    Pure numpy + shapely 2 (vectorised contains_xy); no GDAL."""
    import shapely

    out = np.full((spec.height, spec.width), fill, dtype=np.float32)
    for geom, val in zip(geoms, values):
        if geom is None or geom.is_empty or val is None or not np.isfinite(val):
            continue
        gx0, gy0, gx1, gy1 = geom.bounds
        c0 = max(int(np.floor((gx0 - spec.minx) / spec.res)), 0)
        c1 = min(int(np.ceil((gx1 - spec.minx) / spec.res)), spec.width)
        r0 = max(int(np.floor((spec.maxy - gy1) / spec.res)), 0)
        r1 = min(int(np.ceil((spec.maxy - gy0) / spec.res)), spec.height)
        if c0 >= c1 or r0 >= r1:
            continue
        X, Y = spec.centres(r0, r1, c0, c1)
        inside = shapely.contains_xy(geom, X, Y)
        win = out[r0:r1, c0:c1]
        win[inside] = np.maximum(win[inside], val) if merge == "max" else val
    return out


def impute_missing_heights(geoms, heights, radius_m: float = 50.0):
    """Fill missing heights with the median of known buildings whose centroid lies within
    radius_m (else the global median). Returns (heights, imputed_mask)."""
    h = np.array([np.nan if v is None else float(v) for v in heights], dtype=float)
    miss = ~np.isfinite(h)
    if not miss.any():
        return h, miss
    known = ~miss
    if not known.any():
        raise ValueError("no building has a known height; cannot impute")
    cen = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
    for i in np.flatnonzero(miss):
        d = np.hypot(*(cen[known] - cen[i]).T)
        near = h[known][d <= radius_m]
        h[i] = np.median(near) if near.size else np.median(h[known])
    return h, miss


def build_grids(buildings, bounds, resolution_m: float = 1.0, canopy=None, trunk_zone_fraction: float = 0.25,
                crs: str | None = None, height_field: str = "height_m") -> Grids:
    """Rasterise building footprints (+ optional canopy) onto the working grid.

    buildings: iterable of (shapely geometry, height or None) in the grid CRS, or a path to a
               vector file with `height_field` (read with geopandas, reprojected to `crs`).
    bounds:    (minx, miny, maxx, maxy) in the grid CRS.
    canopy:    None, an array already on this grid (height, width), or a raster path
               (resampled with rasterio, only then needed).
    Missing heights are imputed from neighbours (`impute_missing_heights`); the number of
    imputed buildings is stored on the result as `n_imputed`.
    """
    spec = GridSpec.from_bounds(bounds, resolution_m)
    if isinstance(buildings, (str, bytes)) or hasattr(buildings, "__fspath__"):
        import geopandas as gpd

        gdf = gpd.read_file(buildings)
        if crs is not None:
            gdf = gdf.to_crs(crs)
        pairs = list(zip(gdf.geometry, gdf[height_field]))
    else:
        pairs = list(buildings)
    geoms = [g for g, _ in pairs]
    heights, imputed = impute_missing_heights(geoms, [h for _, h in pairs]) if pairs else (np.array([]), np.array([], bool))
    bh = rasterize_polygons(geoms, heights, spec)

    if canopy is None:
        cv = np.zeros((spec.height, spec.width), np.float32)
    elif isinstance(canopy, np.ndarray):
        if canopy.shape != (spec.height, spec.width):
            raise ValueError(f"canopy shape {canopy.shape} != grid {(spec.height, spec.width)}")
        cv = canopy.astype(np.float32)
    else:
        cv = _resample_canopy_raster(canopy, spec, crs)
    g = grids_from_arrays(bh, cv, resolution_m, trunk_zone_fraction)
    g.transform, g.crs = spec, crs
    g.n_imputed = int(imputed.sum())
    return g


def _resample_canopy_raster(path, spec: GridSpec, crs):
    """Bilinear resampling of a canopy GeoTIFF onto the grid (needs rasterio)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    out = np.zeros((spec.height, spec.width), np.float32)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(spec.minx, spec.maxy, spec.res, spec.res), dst_crs=crs or src.crs,
                  resampling=Resampling.bilinear, dst_nodata=0.0)
    return np.nan_to_num(out, nan=0.0)
