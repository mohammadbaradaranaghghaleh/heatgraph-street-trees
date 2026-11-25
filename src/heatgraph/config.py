"""Configuration loading (stdlib only: tomllib ships with Python 3.11)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Patch:
    id: str
    city: str
    kind: str                     # "patch" | "evaluation" | "transfer"
    source: str                   # "3dbag" | "overture"
    lon: float | None = None
    lat: float | None = None
    bbox: tuple[float, float, float, float] | None = None   # WGS84 W,S,E,N (evaluation areas)
    lcz_intended: str = ""
    status: str = "candidate"
    extra: dict = field(default_factory=dict)


def load_toml(name: str) -> dict:
    with open(CONFIG_DIR / name, "rb") as f:
        return tomllib.load(f)


def load_patches(path: Path | None = None) -> tuple[dict, list[Patch]]:
    """Return (meta, patches) from configs/patches.toml."""
    if path is None:
        raw = load_toml("patches.toml")
    else:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    meta = raw.get("meta", {})
    default_source = meta.get("source_default", "3dbag")
    known = {"id", "city", "lon", "lat", "bbox", "lcz_intended", "status", "source"}
    patches: list[Patch] = []
    for kind in ("patch", "evaluation", "transfer"):
        for p in raw.get(kind, []):
            patches.append(Patch(
                id=p["id"],
                city=p["city"],
                kind=kind,
                source=p.get("source", default_source),
                lon=p.get("lon"),
                lat=p.get("lat"),
                bbox=tuple(p["bbox"]) if "bbox" in p else None,
                lcz_intended=p.get("lcz_intended", ""),
                status=p.get("status", "candidate"),
                extra={k: v for k, v in p.items() if k not in known},
            ))
    ids = [p.id for p in patches]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate patch ids: {sorted(dupes)}")
    return meta, patches
