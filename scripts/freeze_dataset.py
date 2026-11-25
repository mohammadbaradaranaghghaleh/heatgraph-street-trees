"""Freeze the ingested raw dataset: record SHA-256 of every file of every patch, the patch
config and software versions in data/raw/FROZEN.json, and mark patches 'frozen' in
configs/patches.toml. Stage 4 generation refuses to run on files that changed afterwards.

    .\\.venv\\Scripts\\python.exe scripts/freeze_dataset.py
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CFG = ROOT / "configs" / "patches.toml"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from heatgraph.config import load_patches

    _, patches = load_patches()
    wanted = [p for p in patches if p.kind in ("patch", "transfer")]
    out = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "patches_toml_sha256": sha(CFG), "patches": {}}
    for p in wanted:
        d = RAW / p.id
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if man["errors"]:
            raise SystemExit(f"{p.id} has ingestion errors {man['errors']}; not freezing")
        files = {str(f.relative_to(RAW)).replace("\\", "/"): sha(f)
                 for f in sorted(d.rglob("*")) if f.is_file() and f.name != "manifest.json"}
        out["patches"][p.id] = {"city": p.city, "kind": p.kind, "source": p.source, "lon": p.lon, "lat": p.lat,
                                "lcz_intended": p.lcz_intended, "n_files": len(files), "files": files}
    (RAW / "FROZEN.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    text = CFG.read_text(encoding="utf-8")
    for pid in out["patches"]:                       # only patches actually frozen, not evaluation areas
        text = re.sub(rf'(id = "{re.escape(pid)}"[^\[]*?status = )"candidate"', r'\1"frozen"', text, flags=re.S)
    CFG.write_text(text, encoding="utf-8")
    out["patches_toml_sha256_after"] = sha(CFG)
    (RAW / "FROZEN.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"frozen {len(out['patches'])} patches, {sum(v['n_files'] for v in out['patches'].values())} files")


if __name__ == "__main__":
    main()
