"""Revision experiments requested by the simulated LUP review (R2, R3, R4).

    .\\.venv\\Scripts\\python.exe scripts/12_revision_experiments.py --part verify   # R4, seconds, CPU
    .\\.venv\\Scripts\\python.exe scripts/12_revision_experiments.py --part bench    # R2 + R3, GPU, ~30 min

R2  Practitioner heuristics, no surrogate and no optimiser. Each candidate slot takes the score of
    its nearest street segment:
        score_e = flow_e x sunlit_e x (1 - canopy_e)            ("sun x flow" heuristic)
        score_e =          sunlit_e x (1 - canopy_e)            ("sun only" heuristic)
    sunlit_e is the building-only sunlit share of the segment's receptors averaged over the 36 hours,
    canopy_e the share of its receptors under existing canopy >= 2 m, flow_e the gravity flow. Trees
    are placed one at a time on the slot maximising score_e / (1 + trees already on segment e), so
    planting spreads along the priority streets instead of filling one street first.
R3  NSGA-III on the GNN WITHOUT greedy seeding: same population (100), generations (150), operators
    and seed as Stage 5, but the initial population is the empty design plus random feasible designs
    (tree count uniform in 1..B). The engine-verified hypervolume of its front is compared with the
    seeded front saved by 08_benchmark.py, in the normalised space (dose / dose_none, Theil, trees / K).
R4  SOLWEIG-GPU verification on a COMMON mask. 09_solweig_verify.py averaged dUTCI over each model's
    own affected segment-hours (dU < -0.05 C), so the means were not comparable across models. Here
    every model is evaluated on the segment-hours SOLWEIG marks as affected, plus spread and peak.

Outputs: data/stage5/<patch>/revision_r2r3.json, data/stage5/revision_r4_verification.json
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from heatgraph.config import DATA_DIR  # noqa: E402

AFFECTED_C = -0.05


# --------------------------------------------------------------------------- R4
def part_verify(pid="rot_centrum"):
    z = np.load(DATA_DIR / "stage5" / pid / "solweig_verification_arrays.npz", allow_pickle=True)
    names = list(z["names"])
    U = {"solweig": z["U_solweig"].astype(float), "engine": z["U_engine"].astype(float),
         "gnn": z["U_gnn"].astype(float)}
    out = {"patch": pid, "affected_threshold_c": AFFECTED_C,
           "mask": "segment-hours where SOLWEIG dUTCI < -0.05 C (common to all models)", "designs": {}}
    for i, n in enumerate(names[1:], 1):
        dU = {m: u[i] - u[0] for m, u in U.items()}
        mask = dU["solweig"] < AFFECTED_C
        strong = dU["solweig"] < -1.0
        d = {"trees": int(z["X"][i].sum()), "n_segment_hours": int(mask.size),
             "n_affected_solweig_mask": int(mask.sum())}
        for m in ("solweig", "engine", "gnn"):
            d[m] = {"mean_dUTCI_common_mask": float(dU[m][mask].mean()),
                    "mean_dUTCI_all": float(dU[m].mean()),
                    "n_affected_own": int((dU[m] < AFFECTED_C).sum()),
                    "peak_cooling": float(dU[m].min())}
            if m != "solweig":
                e = dU[m][mask] - dU["solweig"][mask]
                d[m].update({"bias_common_mask": float(e.mean()), "mae_common_mask": float(np.abs(e).mean()),
                             "bias_strong_cooling": float((dU[m][strong] - dU["solweig"][strong]).mean())
                             if strong.any() else 0.0,
                             "bias_weak_cooling": float((dU[m][mask & ~strong] - dU["solweig"][mask & ~strong]).mean())
                             if (mask & ~strong).any() else 0.0,
                             "spread_ratio_vs_solweig": float((dU[m] < AFFECTED_C).sum() / max(mask.sum(), 1))})
        out["designs"][n] = d
    p = DATA_DIR / "stage5" / "revision_r4_verification.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"R4 -> {p}")
    for n, d in out["designs"].items():
        print(f"  {n:12s} trees {d['trees']:3d} | common-mask mean dU  SOL {d['solweig']['mean_dUTCI_common_mask']:+.2f}"
              f"  ENG {d['engine']['mean_dUTCI_common_mask']:+.2f}  GNN {d['gnn']['mean_dUTCI_common_mask']:+.2f}"
              f" | bias strong ENG {d['engine']['bias_strong_cooling']:+.2f} weak {d['engine']['bias_weak_cooling']:+.2f}"
              f" | spread ENG/SOL {d['engine']['spread_ratio_vs_solweig']:.2f}")
    return out


# --------------------------------------------------------------------------- R2
def heuristic_path(scene, sg, flow_e, budget, use_flow: bool):
    E, K = scene.n_segments, scene.n_slots
    sun = sg.seg_hourly[:, :, 0].mean(0).astype(float)
    rc = np.vstack(scene.receptors).astype(int)
    can = (scene.base_canopy[rc[:, 0], rc[:, 1]] >= 2.0).astype(float)
    n_rec = np.maximum(np.bincount(sg.seg_of_receptor, minlength=E), 1)
    canopy = np.bincount(sg.seg_of_receptor, weights=can, minlength=E) / n_rec
    seg_score = sun * (1.0 - canopy) * (np.asarray(flow_e, float) if use_flow else 1.0)
    dist = sg.link_attr[0][:, 0]
    slot_seg = np.full(K, -1)
    best = np.full(K, np.inf)
    for j, (k, e) in enumerate(zip(sg.link_slot, sg.link_seg)):
        if dist[j] < best[k]:
            best[k], slot_seg[k] = dist[j], e
    base = np.where(slot_seg >= 0, seg_score[np.maximum(slot_seg, 0)], 0.0)
    x, on_seg, path = np.zeros(K, int), np.zeros(E), []
    for _ in range(int(min(budget, K))):
        pri = np.where(slot_seg >= 0, base / (1.0 + on_seg[np.maximum(slot_seg, 0)]), 0.0)
        pri[x == 1] = -np.inf
        k = int(np.argmax(pri))
        x[k] = 1
        if slot_seg[k] >= 0:
            on_seg[slot_seg[k]] += 1
        path.append(x.copy())
    return np.array(path, int)


def run_patch(pid, gnn, cps, cfg, bm):
    import torch  # noqa: F401
    from heatgraph.engine.receptor_engine import ReceptorEngine
    from heatgraph.exposure_setup import districts, floor_area_population, segment_flows
    from heatgraph.optimize.allocation import SurrogateEvaluator, compare_fronts, run_nsga3
    from heatgraph.realpatch import load_patch
    from heatgraph.realscene import build_scene
    from heatgraph.surrogate.graphs import SceneGraph

    t0 = time.time()
    p = load_patch(pid, cfg)
    s = build_scene(p)
    U0 = np.load(DATA_DIR / "processed" / pid / "labels.npz")["utci"][0]
    sg = SceneGraph(s, baseline_utci=U0, cfg=cfg, device=bm.DEV)
    H, E, K = sg.hours, s.n_segments, s.n_slots
    flow_e = segment_flows(pid, s)
    flows = np.tile(flow_e, (H, 1))
    groups, pop = districts(s, p.core_mask), floor_area_population(s, p.core_mask)
    args = (flows, s.segment_length_m, np.ones(E), groups, pop, 26.0, "theil")
    ev_g = SurrogateEvaluator(gnn, sg, *args[:5], conformal=cps["gnn"], hour_groups=list(range(H)))
    budgets = sorted({max(1, int(round(f * K))) for f in bm.FRACS})
    B = budgets[-1]

    saved = json.loads((DATA_DIR / "stage5" / pid / "results.json").read_text())
    assert saved["n_slots"] == K and saved["budgets"] == budgets, "scene differs from Stage 5"

    # R2 heuristics
    paths = {"heur_sunflow": heuristic_path(s, sg, flow_e, B, True),
             "heur_sun": heuristic_path(s, sg, flow_e, B, False)}
    # R3 unseeded NSGA-III (random feasible initial population)
    rng = np.random.default_rng(0)
    init = [np.zeros(K, int)]
    while len(init) < 100:
        x = np.zeros(K, int)
        x[rng.choice(K, int(rng.integers(1, B + 1)), replace=False)] = 1
        init.append(x)
    t = time.time()
    nu = run_nsga3(ev_g, B, pop_size=100, n_gen=150, seed=0, initial=np.array(init, bool))
    t_nsga = time.time() - t

    designs = {"none": np.zeros((1, K), int)}
    for b in budgets:
        for m, path in paths.items():
            designs[f"{m}_{b}"] = path[b - 1][None]
        designs[f"nsga_unseeded_{b}"] = bm.pick(nu["archive_X"], nu["archive_F"], b)[None]
    front = nu["X"][nu["X"].sum(1) <= B]
    if len(front) > 300:
        order = np.argsort(front.sum(1), kind="stable")
        front = front[order[np.linspace(0, len(front) - 1, 300).round().astype(int)]]

    eng = ReceptorEngine(s.building_heights, np.vstack(s.receptors), s.weather, s.lat, s.lon, cfg, device=bm.DEV)
    seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
    names = list(designs)
    allX = np.vstack([designs[n] for n in names] + [front])
    dose, theil, _ = bm.engine_eval(eng, s, seg, allX, args)
    d0 = float(dose[0])
    assert abs(d0 - saved["engine_dose_none"]) <= 1e-6 * max(1.0, abs(d0)), "engine baseline differs from Stage 5"

    rows = []
    for i, n in enumerate(names[1:], 1):
        method, b = n.rsplit("_", 1)
        rows.append({"design": n, "method": method, "budget": int(b), "trees": int(allX[i].sum()),
                     "engine_dose": float(dose[i]), "engine_theil": float(theil[i]),
                     "engine_dose_reduction_pct": 100 * (d0 - dose[i]) / d0})
    nf = len(names)
    F_unseeded = np.column_stack([dose[nf:] / d0, theil[nf:], allX[nf:].sum(1) / K])
    sf = saved["front_gnn"]
    F_seeded = np.column_stack([[r["engine_dose"] / d0 for r in sf], [r["engine_theil"] for r in sf],
                                [r["trees"] / K for r in sf]])
    gr = [r for r in saved["rows"] if r["method"] == "greedy"]
    F_greedy = np.array([[r["engine_dose"] / d0, r["engine_theil"], r["trees"] / K] for r in gr])
    hv = compare_fronts({"seeded": F_seeded, "unseeded": F_unseeded, "greedy_budgets": F_greedy})
    hv["ref_point"] = [float(v) for v in hv["ref_point"]]

    ref = {(r["method"], r["budget"]): r["engine_dose_reduction_pct"] for r in saved["rows"]
           if r["method"] in ("greedy", "nsga_gnn")}
    rnd = {}
    for r in saved["rows"]:
        if r["method"] == "random":
            rnd.setdefault(r["budget"], []).append(r["engine_dose_reduction_pct"])
    for r in rows:
        r["greedy_pct"] = ref[("greedy", r["budget"])]
        r["nsga_seeded_pct"] = ref[("nsga_gnn", r["budget"])]
        r["random_mean_pct"] = float(np.mean(rnd[r["budget"]]))
        r["share_of_greedy_pct"] = 100 * r["engine_dose_reduction_pct"] / r["greedy_pct"]

    res = {"patch": pid, "n_slots": K, "budgets": budgets, "engine_dose_none": d0, "rows": rows,
           "hypervolume_engine_normalised": hv, "n_front_unseeded": int(len(front)),
           "evals_unseeded": int(nu["n_evals"]), "seconds": {"nsga_unseeded": round(t_nsga, 1),
                                                             "total": round(time.time() - t0, 1)},
           "notes": "heuristics: score = [flow x] building-sunlit share x (1 - canopy share) of the nearest "
                    "segment, diminishing 1/(1+n) per segment; unseeded NSGA-III: random feasible init"}
    (DATA_DIR / "stage5" / pid / "revision_r2r3.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n{pid}: {res['seconds']['total']:.0f} s, HV seeded {hv['seeded']:.4f} unseeded {hv['unseeded']:.4f} "
          f"greedy {hv['greedy_budgets']:.4f}")
    for b in budgets:
        line = "  ".join(f"{r['method']}: {r['engine_dose_reduction_pct']:5.2f}%" for r in rows if r["budget"] == b)
        print(f"  b={b:3d} greedy {ref[('greedy', b)]:5.2f}%  {line}", flush=True)
    return res


def part_bench(patches):
    import torch  # noqa: F401
    from heatgraph.config import load_toml
    from heatgraph.realscene import load_scene
    from heatgraph.surrogate.graphs import SceneGraph
    from heatgraph.surrogate.io import load_gnn

    bm = importlib.import_module("08_benchmark")
    cfg = load_toml("engine.toml")
    probe = load_scene(DATA_DIR / "processed" / bm.VAL[0])
    U = np.load(DATA_DIR / "processed" / bm.VAL[0] / "labels.npz")["utci"]
    sg0 = SceneGraph(probe, baseline_utci=U[0], cfg=cfg, device=bm.DEV)
    gnn = load_gnn(DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt", sg0.build(np.zeros(probe.n_slots), 0), bm.DEV)
    cps = bm.calibrate(gnn, cfg)
    saved = json.loads((DATA_DIR / "stage5" / "conformal.json").read_text())
    assert abs(cps["gnn"].q - saved["gnn"]["q_all"]) < 1e-9, "conformal calibration differs from Stage 5"
    for pid in patches:
        run_patch(pid, gnn, cps, cfg, bm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["verify", "bench", "all"], default="all")
    ap.add_argument("--patch", action="append")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.part in ("verify", "all"):
        part_verify()
    if a.part in ("bench", "all"):
        part_bench(a.patch or ["rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
