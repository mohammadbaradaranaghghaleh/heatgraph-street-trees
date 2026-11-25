"""Block I: engine vs SOLWEIG-GPU on real patches, calibration, and the 3 K acceptance gate.

    .\\.venv\\Scripts\\python.exe scripts/02_solweig_benchmark.py --patch utr_binnenstad --day 2019-07-25
    .\\.venv\\Scripts\\python.exe scripts/02_solweig_benchmark.py --calibrate utr_binnenstad:2019-07-25 \\
        --evaluate ams_grachtengordel:2022-07-19 rot_oude_westen:2020-08-08

Per patch-day: prepare SOLWEIG inputs -> run SOLWEIG-GPU in the `solweig-ref` conda env
(skipped if outputs exist) -> run our engine on the same grid and forcing -> compare Tmrt on
ground pixels of the 500 m core. Calibration fits the [tune] parameters on the calibration
patch-day ONLY and reports the gate on the held-out patch-days (never tuned on).
Results: data/validation/<patch>/<day>/report.json and data/validation/calibration.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from heatgraph.config import DATA_DIR, load_toml  # noqa: E402
from heatgraph.engine import radiation, shadow, sun  # noqa: E402
from heatgraph.engine import svf as svf_mod  # noqa: E402
from heatgraph.realpatch import compare_tmrt, load_patch, local_day_rows, write_solweig_inputs  # noqa: E402

CONDA = Path(os.environ.get("USERPROFILE", "")) / "miniforge3" / "Scripts" / "conda.exe"
GRID = {"ground_warming_k_per_kwm2": [5, 10, 15, 20, 25, 30],
        "wall_warming_k_per_kwm2": [0, 4, 8, 12],
        "albedo_ground": [0.10, 0.15, 0.20, 0.25, 0.30]}


def run_reference(in_dir: Path, day: str) -> Path:
    tif = next(in_dir.glob("output_folder/*/TMRT_*.tif"), None)
    if tif is None:
        cmd = [str(CONDA), "run", "--no-capture-output", "-n", "solweig-ref", "python",
               str(ROOT / "scripts" / "solweig_ref_run.py"), str(in_dir), day]
        with open(in_dir / "solweig_log.txt", "w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        tif = next(in_dir.glob("output_folder/*/TMRT_*.tif"))
    return tif


def read_stack(tif: Path, shape) -> np.ndarray:
    import rasterio

    with rasterio.open(tif) as s:
        a = s.read().astype(np.float32)
    if a.shape[1:] != tuple(shape):
        raise ValueError(f"reference raster {a.shape[1:]} != grid {shape}")
    return a


def engine_parts(p, rows, cfg):
    """Expensive, parameter-independent parts: SVF once, sunlit fraction per hour."""
    import pandas as pd

    g = p.grids
    sky = svf_mod.compute(g.dsm, g.cdsm, g.resolution_m, cfg["svf"]["n_azimuths"], cfg["svf"]["max_distance_m"],
                          cfg["shadow"]["tree_transmissivity"])
    off = pd.Timedelta(minutes=cfg["sun"].get("time_offset_min", 0))
    pos = sun.positions(rows.index + off, p.patch.lat, p.patch.lon)
    dni, dhi, _ = sun.split_radiation(rows["shortwave_radiation"].to_numpy(), pos["zenith"].to_numpy(),
                                      rows.index + off, rows["direct_normal_irradiance"], rows["diffuse_radiation"])
    sunlit = []
    for el, az in zip(pos["elevation"], pos["azimuth"]):
        if el > cfg["sun"]["min_elevation_deg"]:
            sunlit.append(shadow.cast(g.dsm, g.cdsm, g.tdsm, az, el, g.resolution_m, cfg["grid"]["buffer_m"],
                                      cfg["shadow"]["tree_transmissivity"])[0])
        else:
            sunlit.append(np.zeros_like(g.dsm))
    return {"svf": sky["svf"], "sunlit": np.stack(sunlit), "dni": dni, "dhi": dhi,
            "elev": pos["elevation"].to_numpy(), "ta": rows["temperature_2m"].to_numpy(),
            "rh": rows["relative_humidity_2m"].to_numpy()}


def engine_tmrt(parts, params):
    return np.stack([radiation.tmrt(parts["sunlit"][i], parts["svf"], parts["ta"][i], parts["rh"][i],
                                    parts["dni"][i], parts["dhi"][i], parts["elev"][i], params)
                     for i in range(len(parts["ta"]))]).astype(np.float32)


def prepare(patch_id, day, cfg):
    p = load_patch(patch_id, cfg)
    in_dir = DATA_DIR / "validation" / patch_id / day
    write_solweig_inputs(p, in_dir, day)
    ref = read_stack(run_reference(in_dir, day), p.grids.dsm.shape)
    rows = local_day_rows(p.weather, day)
    if len(rows) != ref.shape[0]:
        raise ValueError(f"{len(rows)} forcing rows vs {ref.shape[0]} reference bands")
    parts = engine_parts(p, rows, cfg)
    mask = p.core_mask & ~p.grids.building_mask
    return p, in_dir, ref, parts, mask


def evaluate(p, ref, parts, mask, params, max_mae):
    ours = engine_tmrt(parts, params)
    day = parts["elev"] > 5
    rep = {"all_hours": compare_tmrt(ours, ref, mask, parts["sunlit"], p.grids.cdsm > 0),
           "daytime": compare_tmrt(ours[day], ref[day], mask, parts["sunlit"][day], p.grids.cdsm > 0)}
    rep["gate"] = {"max_tmrt_mae_k": max_mae, "mae_all": rep["all_hours"]["mae"],
                   "bias_all": rep["all_hours"]["bias"],
                   "pass": bool(rep["all_hours"]["mae"] <= max_mae and abs(rep["all_hours"]["bias"]) <= 1.0)}
    return rep, ours


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch"); ap.add_argument("--day")
    ap.add_argument("--calibrate", help="patch:day used for fitting [tune] parameters")
    ap.add_argument("--evaluate", nargs="*", default=[], help="held-out patch:day pairs")
    a = ap.parse_args(argv)
    cfg = load_toml("engine.toml")
    max_mae = cfg["validation"]["max_tmrt_mae_k"]
    base = {k: v for k, v in cfg["radiation"].items() if not isinstance(v, str)}

    if a.patch:
        p, in_dir, ref, parts, mask = prepare(a.patch, a.day, cfg)
        rep, _ = evaluate(p, ref, parts, mask, base, max_mae)
        rep["params"] = base
        (in_dir / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep["gate"]), "| daytime MAE", round(rep["daytime"]["mae"], 2),
              "bias", round(rep["daytime"]["bias"], 2))
        return 0

    cal_id, cal_day = a.calibrate.split(":")
    p, _, ref, parts, mask = prepare(cal_id, cal_day, cfg)
    before, _ = evaluate(p, ref, parts, mask, base, max_mae)
    best = None
    for combo in itertools.product(*GRID.values()):
        params = {**base, **dict(zip(GRID, combo))}
        r, _ = evaluate(p, ref, parts, mask, params, max_mae)
        score = r["all_hours"]["mae"]
        if best is None or score < best[0]:
            best = (score, params, r)
    out = {"calibration": {"patch_day": a.calibrate, "before": before["gate"], "after": best[2]["gate"],
                           "params": {k: best[1][k] for k in GRID}}, "held_out": {}}
    print("calibration", a.calibrate, "MAE", round(before["gate"]["mae_all"], 2), "->", round(best[0], 2),
          {k: best[1][k] for k in GRID})
    for pd_ in a.evaluate:
        pid, day = pd_.split(":")
        q, in_dir, ref2, parts2, mask2 = prepare(pid, day, cfg)
        r0, _ = evaluate(q, ref2, parts2, mask2, base, max_mae)
        r1, _ = evaluate(q, ref2, parts2, mask2, best[1], max_mae)
        out["held_out"][pd_] = {"default": r0["gate"], "calibrated": r1["gate"], "daytime_calibrated": r1["daytime"]}
        (in_dir / "report.json").write_text(json.dumps({"default": r0, "calibrated": r1}, indent=2), encoding="utf-8")
        print("held-out", pd_, "MAE default", round(r0["gate"]["mae_all"], 2), "calibrated",
              round(r1["gate"]["mae_all"], 2), "bias", round(r1["gate"]["bias_all"], 2), "PASS" if r1["gate"]["pass"] else "FAIL")
    (DATA_DIR / "validation" / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
