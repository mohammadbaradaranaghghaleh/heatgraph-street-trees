"""Package the released results (everything `reproduce.py` and `app.py` read) into dist/.

    python scripts/make_release_archive.py                 # build dist/heatgraph-data-<date>.zip (~0.2 GB)
    python scripts/make_release_archive.py --with-rasters  # also dist/heatgraph-rasters-<date>.zip (~5.4 GB)
    python scripts/make_release_archive.py --verify        # check extracted data/ against data/RELEASE_MANIFEST.json

The data archive holds the frozen-dataset hash record, ingestion QA table, processed scenes and
counterfactual labels of all 18 patches, trained surrogates and metrics, engine calibration, and the
Stage 5 benchmark / Pareto / SOLWEIG-verification results. The raster archive adds the SOLWEIG-GPU
input and output GeoTIFFs (regenerable with scripts/02 and scripts/09). Raw downloads (data/raw/<patch>)
are not redistributed: they are re-fetched from 3DBAG, Overture, Meta/WRI canopy and Open-Meteo by
scripts/01_ingest.py and checked against data/raw/FROZEN.json. Inside each archive,
data/RELEASE_MANIFEST.json lists the SHA-256 of every file; dist/SHA256SUMS.txt covers the archives.
Reads data/ only; writes only to dist/ (or the extracted manifest with --verify, read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIST = ROOT / "dist"
SMALL = {".json", ".txt", ".csv"}          # provenance files kept from raster folders


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_raster_tree(rel: Path) -> bool:
    parts = rel.parts           # relative to data/
    return (parts[0] == "validation" and len(parts) > 2) or (parts[0] == "stage5" and "solweig" in parts[2:3])


def select(with_rasters: bool) -> tuple[list[Path], list[Path]]:
    core, rasters = [], []
    for f in (DATA / "raw").glob("*"):
        if f.is_file() and f.name != ".gitkeep":
            core.append(f)                                   # FROZEN.json, ingest_summary.csv, logs
    for sub in ("processed", "models", "validation", "stage5"):
        for f in sorted((DATA / sub).rglob("*")):
            if not f.is_file() or f.name == ".gitkeep":
                continue
            rel = f.relative_to(DATA)
            if is_raster_tree(rel) and f.suffix.lower() not in SMALL:
                if with_rasters:
                    rasters.append(f)
                continue
            core.append(f)
    return sorted(core), sorted(rasters)


def write_zip(path: Path, files: list[Path], manifest_name: str) -> dict:
    manifest = {"created": time.strftime("%Y-%m-%d"), "files": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in files:
            arc = f.relative_to(ROOT).as_posix()
            manifest["files"][arc] = {"sha256": sha256(f), "bytes": f.stat().st_size}
            z.write(f, arc)
        z.writestr(f"data/{manifest_name}", json.dumps(manifest, indent=1))
    return manifest


def verify() -> int:
    bad = 0
    for name in ("RELEASE_MANIFEST.json", "RELEASE_MANIFEST_RASTERS.json"):
        m = DATA / name
        if not m.exists():
            continue
        files = json.loads(m.read_text(encoding="utf-8"))["files"]
        for arc, info in files.items():
            p = ROOT / arc
            if not p.exists() or sha256(p) != info["sha256"]:
                print(f"MISMATCH {arc}")
                bad += 1
        print(f"{name}: {len(files)} files checked, {bad} mismatches")
    if not (DATA / "RELEASE_MANIFEST.json").exists():
        print("data/RELEASE_MANIFEST.json not found: extract the data archive at the repository root first")
        return 2
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-rasters", action="store_true", help="also build the SOLWEIG raster archive")
    ap.add_argument("--verify", action="store_true", help="verify extracted files against the manifest")
    a = ap.parse_args(argv)
    if a.verify:
        return verify()

    core, rasters = select(a.with_rasters)
    stamp = time.strftime("%Y%m%d")
    out = [(DIST / f"heatgraph-data-{stamp}.zip", core, "RELEASE_MANIFEST.json")]
    if a.with_rasters:
        out.append((DIST / f"heatgraph-rasters-{stamp}.zip", rasters, "RELEASE_MANIFEST_RASTERS.json"))
    sums = []
    for path, files, mname in out:
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"writing {path.relative_to(ROOT)}: {len(files)} files, {mb:,.0f} MB uncompressed", flush=True)
        write_zip(path, files, mname)
        with zipfile.ZipFile(path) as z:                        # integrity check of the written archive
            if z.testzip() is not None:
                print(f"corrupt member in {path.name}")
                return 1
        sums.append(f"{sha256(path)}  {path.name}")
        print(f"  -> {path.stat().st_size / 1e6:,.0f} MB")
    (DIST / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"checksums: {(DIST / 'SHA256SUMS.txt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
