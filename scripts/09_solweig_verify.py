"""Stage 5b: independent SOLWEIG-GPU verification of optimised tree layouts (rot_centrum).

    .\\.venv\\Scripts\\python.exe scripts/09_solweig_verify.py [--patch rot_centrum]

Five designs are taken from the NSGA-III (GNN) Pareto front using the decision-maker's view
(GNN robust objectives, not the engine): none, low budget (~10 % of slots), knee point,
high budget (~30 %), max cooling (lowest GNN dose). Each is run through SOLWEIG-GPU (the
`solweig-ref` conda env) for the three heatwave days; UTCI at 08-19 local is sampled at the
same receptors and averaged per segment. Compared per design: tree-induced dUTCI (design -
none) from SOLWEIG vs our engine vs the GNN, and the person-weighted heat-dose reduction.
SOLWEIG's absolute UTCI uses its own wind treatment, so comparisons use CHANGES relative to
each model's own no-planting run. Outputs: data/stage5/<patch>/solweig_verification.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from heatgraph.config import DATA_DIR, load_patches, load_toml  # noqa: E402
from heatgraph.exposure_setup import districts, floor_area_population, segment_flows  # noqa: E402
from heatgraph.optimize.allocation import objectives_from_utci  # noqa: E402
from heatgraph.realpatch import load_patch, write_solweig_inputs  # noqa: E402
from heatgraph.realscene import build_scene  # noqa: E402

bench = __import__("02_solweig_benchmark")
TZ = "Europe/Amsterdam"


def select_designs(front_X, front_rows, K):
    trees = np.array([r["trees"] for r in front_rows])
    dose = np.array([r["gnn_dose_upper"] for r in front_rows])
    # best (lowest GNN dose) design for each tree count, then pick along that curve
    best = {}
    for i, (t, d) in enumerate(zip(trees, dose)):
        if t > 0 and (t not in best or d < dose[best[t]]):
            best[t] = i
    ts = np.array(sorted(best))
    idx = np.array([best[t] for t in ts])
    near = lambda frac: idx[np.argmin(np.abs(ts - frac * K))]
    x = (ts - ts.min()) / max(np.ptp(ts), 1)
    y = (dose[idx] - dose[idx].min()) / max(np.ptp(dose[idx]), 1e-9)
    knee = idx[np.argmax(np.abs((1 - x) - y) / np.sqrt(2))]       # distance to line (0,1)-(1,0)
    sel = {"none": None, "low_budget": near(0.10), "knee": knee, "high_budget": near(0.30),
           "max_cooling": idx[np.argmin(dose[idx])]}
    return {k: (np.zeros(K, int) if v is None else front_X[v]) for k, v in sel.items()}


def solweig_segment_utci(p, s, design, day, hours_local, workdir):
    import rasterio

    d = workdir / day
    write_solweig_inputs(p, d, day, cdsm=s.canopy_with(design))
    tif = bench.run_reference(d, day)
    utci_tif = next(d.glob("output_folder/*/UTCI_*.tif"))
    with rasterio.open(utci_tif) as src:
        stack = src.read().astype(np.float64)                      # (24 local hours, H, W)
    _ = tif
    rows = []
    for e, rc in enumerate(s.receptors):
        rows.append(stack[hours_local][:, rc[:, 0], rc[:, 1]].mean(1))
    return np.stack(rows, 1)                                       # (len(hours), E)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="rot_centrum")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_toml("engine.toml")
    pid = a.patch
    res = json.loads((DATA_DIR / "stage5" / pid / "results.json").read_text())
    z = np.load(DATA_DIR / "stage5" / pid / "designs.npz")
    p = load_patch(pid, cfg)
    s = build_scene(p)
    K, E = s.n_slots, s.n_segments
    designs = select_designs(z["front"], res["front_gnn"], K)
    meta, _ = load_patches()
    days = meta["heatwave_days"]
    loc = s.weather.index.tz_convert(TZ)
    hours_local = sorted(set(loc.hour))

    # GNN + engine predictions for the same designs (re-computed, not looked up)
    import torch

    from heatgraph.engine.receptor_engine import ReceptorEngine, segment_mean
    from heatgraph.surrogate.graphs import SceneGraph
    from heatgraph.surrogate.io import load_gnn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U0 = np.load(DATA_DIR / "processed" / pid / "labels.npz")["utci"][0]
    sg = SceneGraph(s, baseline_utci=U0, cfg=cfg, device=dev)
    gnn = load_gnn(DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt", sg.build(np.zeros(K), 0), dev)
    eng = ReceptorEngine(s.building_heights, np.vstack(s.receptors), s.weather, s.lat, s.lon, cfg, device=dev)
    seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
    names = list(designs)
    X = np.stack([designs[n] for n in names])
    with torch.no_grad():
        U_gnn = gnn.predict(sg.build_batch(X).to(dev)).cpu().numpy().reshape(len(X), sg.hours, E)
    U_eng = np.concatenate([segment_mean(eng.utci(np.stack([s.canopy_with(x)]))["utci"], seg, E) for x in X])

    # SOLWEIG-GPU, one run per design and day, in the scene's (UTC-ordered) hour order
    U_sol = np.zeros_like(U_eng)
    day_of_row = loc.strftime("%Y-%m-%d")
    for i, n in enumerate(names):
        for day in days:
            rows = np.flatnonzero(day_of_row == day)
            seg_utci = solweig_segment_utci(p, s, X[i], day, [loc[r].hour for r in rows],
                                            DATA_DIR / "stage5" / pid / "solweig" / n)
            U_sol[i, rows] = seg_utci
            print(f"  SOLWEIG {n:12s} {day} done", flush=True)

    flows = np.tile(segment_flows(pid, s), (sg.hours, 1))
    args = (flows, s.segment_length_m, np.ones(E), districts(s, p.core_mask), floor_area_population(s, p.core_mask),
            26.0, "theil")
    out = {"patch": pid, "designs": {}, "notes": "changes are relative to each model's own no-planting run"}
    for model, U in (("solweig", U_sol), ("engine", U_eng), ("gnn", U_gnn)):
        dose, theil, _ = objectives_from_utci(U, *args)
        for i, n in enumerate(names):
            d = out["designs"].setdefault(n, {"trees": int(X[i].sum())})
            dU = U[i] - U[0]
            d[model] = {"dose": float(dose[i]), "dose_reduction_pct": float(100 * (dose[0] - dose[i]) / dose[0]),
                        "theil": float(theil[i]), "mean_dUTCI": float(dU.mean()),
                        "mean_dUTCI_affected": float(dU[dU < -0.05].mean()) if (dU < -0.05).any() else 0.0,
                        "max_cooling": float(dU.min())}
    for i, n in enumerate(names[1:], 1):
        ds, de, dg = (U_sol[i] - U_sol[0]).ravel(), (U_eng[i] - U_eng[0]).ravel(), (U_gnn[i] - U_gnn[0]).ravel()
        out["designs"][n]["agreement"] = {
            "corr_engine_vs_solweig": float(np.corrcoef(de, ds)[0, 1]), "corr_gnn_vs_solweig": float(np.corrcoef(dg, ds)[0, 1]),
            "mae_engine_vs_solweig": float(np.abs(de - ds).mean()), "mae_gnn_vs_solweig": float(np.abs(dg - ds).mean())}
    np.savez_compressed(DATA_DIR / "stage5" / pid / "solweig_verification_arrays.npz", names=np.array(names), X=X,
                        U_solweig=U_sol, U_engine=U_eng, U_gnn=U_gnn)
    (DATA_DIR / "stage5" / pid / "solweig_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'design':12s} trees | dose reduction %: SOLWEIG  engine   GNN | mean dUTCI (affected) SOLWEIG/engine/GNN")
    for n in names:
        d = out["designs"][n]
        print(f"{n:12s} {d['trees']:5d} | {d['solweig']['dose_reduction_pct']:8.1f} {d['engine']['dose_reduction_pct']:7.1f} "
              f"{d['gnn']['dose_reduction_pct']:6.1f} | {d['solweig']['mean_dUTCI_affected']:+.2f} / "
              f"{d['engine']['mean_dUTCI_affected']:+.2f} / {d['gnn']['mean_dUTCI_affected']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
