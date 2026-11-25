"""Stage 4a: counterfactual tree-planting dataset on real patches (GPU receptor engine).

    .\\.venv\\Scripts\\python.exe scripts/04_generate_dataset.py                  # all frozen patches
    .\\.venv\\Scripts\\python.exe scripts/04_generate_dataset.py --patch utr_binnenstad --n 20

Per patch: real scene (segments, receptors, slots) -> N plantings (index 0 = no new trees,
i.e. the existing canopy; others: slot density ~ U(0.02, 0.40), seeded by patch id) ->
calibrated engine (engine.toml) on the GPU -> segment-mean UTCI and Tmrt per (planting, hour).
Outputs: data/processed/<patch>/{scene.npz, scene.json, weather.csv, labels.npz, meta.json}.
Refuses to run unless data/raw/FROZEN.json lists the patch with unchanged file hashes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR, load_toml  # noqa: E402
from heatgraph.engine.receptor_engine import ReceptorEngine, segment_mean  # noqa: E402
from heatgraph.realpatch import load_patch  # noqa: E402
from heatgraph.realscene import build_scene, save_scene  # noqa: E402

DENSITY = (0.02, 0.40)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(pid: str) -> dict:
    frozen = json.loads((DATA_DIR / "raw" / "FROZEN.json").read_text(encoding="utf-8"))
    entry = frozen["patches"].get(pid)
    if entry is None:
        raise SystemExit(f"{pid} is not in the frozen dataset")
    for rel, digest in entry["files"].items():
        if sha(DATA_DIR / "raw" / rel) != digest:
            raise SystemExit(f"{rel} changed since the freeze; re-freeze deliberately before generating")
    return entry


def plantings(n_slots: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dens = rng.uniform(*DENSITY, size=n)
    X = (rng.random((n, n_slots)) < dens[:, None]).astype(np.int8)
    X[0] = 0
    return X


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="append")
    ap.add_argument("--n", type=int, default=81, help="plantings per patch incl. the unplanted one")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_toml("engine.toml")
    frozen = json.loads((DATA_DIR / "raw" / "FROZEN.json").read_text(encoding="utf-8"))
    pids = a.patch or list(frozen["patches"])
    for pid in pids:
        t0 = time.time()
        check_frozen(pid)
        p = load_patch(pid, cfg)
        s = build_scene(p)
        out = DATA_DIR / "processed" / pid
        save_scene(s, out)
        rc = np.vstack(s.receptors)
        seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
        eng = ReceptorEngine(s.building_heights, rc, s.weather, s.lat, s.lon, cfg, device=a.device)
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        X = plantings(s.n_slots, a.n, seed)
        U = np.empty((len(X), len(s.weather), s.n_segments), np.float32)
        T = np.empty_like(U)
        for i in range(0, len(X), a.batch):
            cd = np.stack([s.canopy_with(x) for x in X[i:i + a.batch]])
            r = eng.utci(cd)
            U[i:i + a.batch] = segment_mean(r["utci"], seg, s.n_segments)
            T[i:i + a.batch] = segment_mean(r["tmrt"], seg, s.n_segments)
        if np.isnan(U).any():
            raise SystemExit(f"{pid}: NaN UTCI in labels ({int(np.isnan(U).sum())} values)")
        np.savez_compressed(out / "labels.npz", plantings=X, utci=U, tmrt=T)
        meta = {"patch": pid, "n_plantings": len(X), "n_slots": s.n_slots, "n_segments": s.n_segments,
                "n_receptors": int(len(rc)), "hours": len(s.weather), "density_range": DENSITY, "seed": seed,
                "device": a.device, "seconds": round(time.time() - t0, 1), "engine_cfg": cfg,
                "frozen_at": frozen["frozen_at_utc"],
                "mean_tree_effect_c": float((U[1:] - U[:1]).mean()),
                "max_tree_cooling_c": float((U[1:] - U[:1]).min())}
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"{pid:24s} seg {s.n_segments:4d} rec {len(rc):5d} slots {s.n_slots:3d} hours {len(s.weather)} "
              f"| mean effect {meta['mean_tree_effect_c']:+.3f} C, max cooling {meta['max_tree_cooling_c']:+.2f} C "
              f"| {meta['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
