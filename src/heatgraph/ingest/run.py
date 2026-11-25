"""Block A orchestrator.

    py -3 -m heatgraph.ingest.run                 # dry-run: list requests + size estimate, no network
    py -3 -m heatgraph.ingest.run --execute --patch ams_grachtengordel   # one patch, real I/O

Dry-run is the default on purpose: nothing touches the network or disk (except
printing) unless --execute is given. Each executed patch gets a manifest.json with
source URLs, file SHA-256 hashes, package versions and an explicit UTC timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import DATA_DIR, load_patches, load_toml
from . import buildings_3dbag, buildings_overture, canopy_meta, network_overture, osm_network, weather_openmeteo

BUILDING_FETCHERS = {"3dbag": buildings_3dbag, "overture": buildings_overture}
# `network` (Overture transportation, OSM-derived) is the default; `osm` (live Overpass) is
# opt-in only, because the public Overpass endpoint asks automated users to run their own instance.
LAYERS = ("buildings", "canopy", "network", "weather")
OPTIONAL_LAYERS = ("osm",)


@dataclass
class Context:
    half_size_m: float
    heatwave_days: list[str]
    raw_dir: Path
    cache_dir: Path
    planned: set = field(default_factory=set)   # shared resources already counted in a dry-run

    def patch_dir(self, patch) -> Path:
        d = self.raw_dir / patch.id
        d.mkdir(parents=True, exist_ok=True)
        return d


def fetchers_for(patch, layers):
    mods = {"buildings": BUILDING_FETCHERS[patch.source], "canopy": canopy_meta,
            "network": network_overture, "osm": osm_network, "weather": weather_openmeteo}
    return [(name, mods[name]) for name in layers]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for pkg in ("geopandas", "shapely", "pyproj", "rasterio", "osmnx", "overturemaps", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def write_manifest(patch, ctx, results: dict, errors: dict) -> Path:
    """Merge into an existing manifest: re-running one layer replaces only that layer's
    files/errors and keeps the record of the other layers."""
    path = ctx.patch_dir(patch) / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rerun = set(results) | set(errors)
    files = {k: v for k, v in old.get("files", {}).items() if v.get("layer") not in rerun}
    errors = {**{k: v for k, v in old.get("errors", {}).items() if k not in rerun}, **errors}
    retrieved = {**old.get("retrieved_at_utc_by_layer", {}),
                 **{layer: datetime.now(timezone.utc).isoformat(timespec="seconds") for layer in rerun}}
    for layer, paths in results.items():
        for p in paths:
            ps = sorted(p.rglob("*")) if p.is_dir() else [p]
            for f in ps:
                if f.is_file():
                    files[str(f.relative_to(ctx.raw_dir))] = {"layer": layer, "sha256": _sha256(f),
                                                              "bytes": f.stat().st_size}
    manifest = {
        "patch": {"id": patch.id, "city": patch.city, "kind": patch.kind, "source": patch.source,
                  "lon": patch.lon, "lat": patch.lat, "lcz_intended": patch.lcz_intended,
                  "status": patch.status},
        "half_size_m": ctx.half_size_m,
        "heatwave_days": ctx.heatwave_days,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieved_at_utc_by_layer": retrieved,
        "software": _versions(),
        "files": files,
        "errors": errors,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HeatGraph Block A: data ingestion")
    ap.add_argument("--execute", action="store_true", help="perform network I/O (default: dry-run)")
    ap.add_argument("--patch", action="append", help="patch id (repeatable); default: all")
    ap.add_argument("--kind", choices=["patch", "transfer"], action="append",
                    help="patch kinds to include (default: patch and transfer)")
    ap.add_argument("--layer", choices=LAYERS + OPTIONAL_LAYERS, action="append",
                    help="layers to fetch (default: buildings, canopy, network, weather)")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    meta, patches = load_patches()
    engine = load_toml("engine.toml")
    ctx = Context(half_size_m=meta.get("core_size_m", 500) / 2 + engine["grid"]["buffer_m"],
                  heatwave_days=meta.get("heatwave_days", []),
                  raw_dir=DATA_DIR / "raw", cache_dir=DATA_DIR / "cache")

    kinds = set(args.kind or ["patch", "transfer"])
    selected = [p for p in patches if p.kind in kinds and (not args.patch or p.id in args.patch)]
    skipped_eval = [p.id for p in patches if p.kind == "evaluation"]
    layers = args.layer or list(LAYERS)
    if args.patch:
        missing = set(args.patch) - {p.id for p in selected}
        if missing:
            print(f"Unknown or filtered-out patch ids: {sorted(missing)}", file=sys.stderr)
            return 2

    mode = "EXECUTE" if args.execute else "DRY-RUN (no network, no files written)"
    print(f"HeatGraph ingestion - {mode}")
    print(f"patches: {len(selected)} | layers: {', '.join(layers)} | "
          f"window: {2 * ctx.half_size_m:.0f} m square (core + buffer)")
    if skipped_eval:
        print(f"note: evaluation areas {skipped_eval} need tiled ingestion (Block A2, not implemented yet)")

    total_mb = 0.0
    failures = 0
    for p in selected:
        print(f"\n[{p.kind}] {p.id}  ({p.city}, {p.source}, {p.lcz_intended}, {p.status})")
        results, errors = {}, {}
        for name, mod in fetchers_for(p, layers):
            if not args.execute:
                for r in mod.plan(p, ctx):
                    total_mb += r.est_mb
                    print(f"  - {name:<9} ~{r.est_mb:>6.1f} MB  {r.url}" + (f"  [{r.note}]" if r.note else ""))
                continue
            try:
                results[name] = mod.fetch(p, ctx)
                print(f"  ok   {name}")
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"
                failures += 1
                print(f"  FAIL {name}: {errors[name]}", file=sys.stderr)
        if args.execute:
            print(f"  manifest -> {write_manifest(p, ctx, results, errors)}")

    if not args.execute:
        print(f"\nEstimated total transfer: ~{total_mb:.0f} MB (rough upper bound). "
              f"Re-run with --execute to fetch.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
