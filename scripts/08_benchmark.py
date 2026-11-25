"""Stage 5: optimisation benchmark on the unseen test cities (Rotterdam x3, Freiburg).

    .\\.venv\\Scripts\\python.exe scripts/08_benchmark.py                 # all test patches
    .\\.venv\\Scripts\\python.exe scripts/08_benchmark.py --patch rot_centrum

Methods (all optimise the robust objective = conformal UPPER bound of the surrogate):
  nsga_gnn     NSGA-III on the v2 Hetero-GNN (primary), seeded with the greedy path + archive
  nsga_linear  NSGA-III on the linear physics surrogate, same seeding
  greedy       greedy on the linear surrogate (dose only), the classical baseline
  random       10 random layouts per budget (exactly b trees)
Budgets: 5, 10, 20, 30, 40 % of the patch's candidate slots. For NSGA methods the design at
budget b is the archive design with <= b trees and the lowest surrogate dose.
Ground truth: every selected design is re-simulated with the calibrated engine (GPU receptor
engine); reported dose reduction is ENGINE dose vs the engine's no-new-trees dose.
Objectives: person-weighted heat dose (UTCI > 26 C, flows from the gravity model), Theil of
per-capita dose across 4 core districts (population PROXY = floor area), number of trees.
Conformal calibration (Mondrian by hour, alpha = 0.1) on the leave-patch-out validation
patches. Outputs: data/stage5/<patch>/{results.json, designs.npz}, data/stage5/conformal.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.config import DATA_DIR, load_toml  # noqa: E402
from heatgraph.engine.receptor_engine import ReceptorEngine, segment_mean  # noqa: E402
from heatgraph.exposure_setup import districts, floor_area_population, segment_flows  # noqa: E402
from heatgraph.optimize.additive import AdditiveEvaluator  # noqa: E402
from heatgraph.optimize.allocation import (SurrogateEvaluator, greedy_path, is_nondominated,  # noqa: E402
                                           objectives_from_utci, run_nsga3)
from heatgraph.realpatch import load_patch  # noqa: E402
from heatgraph.realscene import build_scene, load_scene  # noqa: E402
from heatgraph.surrogate.graphs import SceneGraph  # noqa: E402
from heatgraph.surrogate.io import load_gnn  # noqa: E402
from heatgraph.uq.conformal import SplitConformal  # noqa: E402

TEST = ["rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt"]
VAL = ["ams_bos_en_lommer", "ein_strijp_s", "amf_vathorst"]
FRACS = [0.05, 0.10, 0.20, 0.30, 0.40]
ALPHA, N_RANDOM = 0.1, 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = DATA_DIR / "stage5"


def linear_coef():
    return np.array(json.loads((DATA_DIR / "models" / "metrics_linear_physics.json").read_text())["coef_per_hour"])


def calibrate(gnn, cfg):
    """Mondrian-by-hour split conformal for both surrogates on the validation patches."""
    res = {"gnn": {}, "linear": {}}
    coef = linear_coef()
    for pid in VAL:
        s = load_scene(DATA_DIR / "processed" / pid)
        z = np.load(DATA_DIR / "processed" / pid / "labels.npz")
        X, U = z["plantings"], z["utci"]
        sg = SceneGraph(s, baseline_utci=U[0], cfg=cfg, device=DEV)
        for name in ("gnn", "linear"):
            if name == "gnn":
                ev = SurrogateEvaluator(gnn, sg, None, None, None, None, {0: 1})
            else:
                ev = AdditiveEvaluator(sg, coef, None, None, None, None, {0: 1})
            P = ev.predict_utci(X, "point")
            for h in range(sg.hours):
                res[name].setdefault(h, []).append((U[:, h] - P[:, h]).ravel())
    out = {}
    for name, d in res.items():
        r = np.concatenate([np.concatenate(v) for v in d.values()])
        g = np.concatenate([np.full(sum(len(x) for x in v), h) for h, v in d.items()])
        out[name] = SplitConformal(ALPHA).calibrate(r, np.zeros_like(r), g)
    return out


def engine_eval(eng, scene, seg, X, args):
    """Engine (ground truth) objectives for designs X: dose, theil, per-segment UTCI."""
    U = []
    for i in range(0, len(X), 8):
        cd = np.stack([scene.canopy_with(x) for x in X[i:i + 8]])
        U.append(segment_mean(eng.utci(cd)["utci"], seg, scene.n_segments))
    U = np.concatenate(U)
    dose, theil, _ = objectives_from_utci(U, *args)
    return dose, theil, U


def pick(arch_X, arch_F, b):
    ok = arch_X.sum(1) <= b
    if not ok.any():
        return np.zeros(arch_X.shape[1], int)
    j = np.flatnonzero(ok)[np.argmin(arch_F[ok, 0])]
    return arch_X[j]


def run_patch(pid, gnn, cps, cfg):
    t0 = time.time()
    p = load_patch(pid, cfg)
    s = build_scene(p)
    U0 = np.load(DATA_DIR / "processed" / pid / "labels.npz")["utci"][0]
    sg = SceneGraph(s, baseline_utci=U0, cfg=cfg, device=DEV)
    H, E, K = sg.hours, s.n_segments, s.n_slots
    flows = np.tile(segment_flows(pid, s), (H, 1))
    groups, pop = districts(s, p.core_mask), floor_area_population(s, p.core_mask)
    args = (flows, s.segment_length_m, np.ones(E), groups, pop, 26.0, "theil")
    hg = list(range(H))
    ev_g = SurrogateEvaluator(gnn, sg, *args[:5], conformal=cps["gnn"], hour_groups=hg)
    ev_l = AdditiveEvaluator(sg, linear_coef(), *args[:5], conformal=cps["linear"], hour_groups=hg)
    budgets = sorted({max(1, int(round(f * K))) for f in FRACS})
    B = budgets[-1]

    greedy = greedy_path(ev_l, B)
    # v1 search effort (pop 60 x 40 gen, ~2.4k GNN evaluations) never beat the greedy seeds on
    # rot_centrum; v2 uses ~6x more (evaluations are cheap: ~3.5 ms per design on the RTX 5070).
    t = time.time(); nl = run_nsga3(ev_l, B, pop_size=100, n_gen=300, seed=0, greedy=greedy); tl = time.time() - t
    t = time.time(); ng = run_nsga3(ev_g, B, pop_size=100, n_gen=150, seed=0, greedy=greedy); tg = time.time() - t
    rng = np.random.default_rng(0)
    designs = {"none": np.zeros((1, K), int)}
    for b in budgets:
        designs[f"greedy_{b}"] = greedy[b - 1][None]
        designs[f"nsga_linear_{b}"] = pick(nl["archive_X"], nl["archive_F"], b)[None]
        designs[f"nsga_gnn_{b}"] = pick(ng["archive_X"], ng["archive_F"], b)[None]
        R = np.zeros((N_RANDOM, K), int)
        for r in range(N_RANDOM):
            R[r, rng.choice(K, b, replace=False)] = 1
        designs[f"random_{b}"] = R
    front = ng["X"][ng["X"].sum(1) <= B]
    if len(front) > 300:                     # engine cost: keep <= 300 front designs, stratified by tree count
        order = np.argsort(front.sum(1), kind="stable")
        front = front[order[np.linspace(0, len(front) - 1, 300).round().astype(int)]]

    eng = ReceptorEngine(s.building_heights, np.vstack(s.receptors), s.weather, s.lat, s.lon, cfg, device=DEV)
    seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
    names = list(designs)
    allX = np.vstack([designs[n] for n in names] + [front])
    dose, theil, Ueng = engine_eval(eng, s, seg, allX, args)
    sur_g = ev_g.objectives(allX, "point"); sur_g_up = ev_g.objectives(allX, "upper")
    sur_l = ev_l.objectives(allX, "point")
    d0 = float(dose[0])
    rows, i = [], 0
    for n in names:
        m = len(designs[n])
        method, b = (n.rsplit("_", 1) + ["0"])[:2] if n != "none" else ("none", "0")
        for j in range(m):
            k = i + j
            rows.append({"design": n, "method": method, "budget": int(b), "replicate": j,
                         "trees": int(allX[k].sum()), "engine_dose": float(dose[k]), "engine_theil": float(theil[k]),
                         "engine_dose_reduction_pct": 100 * (d0 - dose[k]) / d0,
                         "gnn_dose_point": float(sur_g[k, 0]), "gnn_dose_upper": float(sur_g_up[k, 0]),
                         "linear_dose_point": float(sur_l[k, 0])})
        i += m
    front_rows = [{"trees": int(allX[i + j].sum()), "engine_dose": float(dose[i + j]),
                   "engine_theil": float(theil[i + j]), "gnn_dose_upper": float(sur_g_up[i + j, 0]),
                   "gnn_theil_upper": float(sur_g_up[i + j, 1])} for j in range(len(front))]
    # equity at matched cooling: best engine Theil on the GNN front among designs with <= b trees
    # that keep >= 90 % of greedy's engine dose reduction at that budget
    ft = np.array([r["trees"] for r in front_rows]); fd = np.array([r["engine_dose"] for r in front_rows])
    fth = np.array([r["engine_theil"] for r in front_rows])
    equity = []
    for b in budgets:
        gr = next(r for r in rows if r["method"] == "greedy" and r["budget"] == b)
        keep = (ft <= b) & (fd <= gr["engine_dose"] + 0.1 * (d0 - gr["engine_dose"]))
        if keep.any():
            j = int(np.argmin(np.where(keep, fth, np.inf)))
            equity.append({"budget": b, "greedy_theil": gr["engine_theil"], "front_theil": float(fth[j]),
                           "theil_change_pct": float(100 * (fth[j] - gr["engine_theil"]) / gr["engine_theil"]),
                           "cooling_kept_pct": float(100 * (d0 - fd[j]) / (d0 - gr["engine_dose"]))})
    res = {"patch": pid, "n_slots": K, "n_segments": E, "hours": H, "budgets": budgets, "engine_dose_none": d0,
           "equity_at_matched_cooling": equity, "engine_theil_none": float(theil[0]),
           "rows": rows, "front_gnn": front_rows, "seconds": {"nsga_linear": round(tl, 1), "nsga_gnn": round(tg, 1),
                                                               "total": round(time.time() - t0, 1)},
           "evals": {"nsga_linear": nl["n_evals"], "nsga_gnn": ng["n_evals"]},
           "population": "PROXY floor area per core quadrant", "flows": "gravity, uniform origins, Overture places"}
    od = OUT / pid
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez_compressed(od / "designs.npz", names=np.array(names), X=allX[:len(allX) - len(front)],
                        front=front, engine_utci=Ueng, slots=s.slots)
    tab = {}
    for r in rows:
        tab.setdefault((r["method"], r["budget"]), []).append(r["engine_dose_reduction_pct"])
    print(f"\n{pid}: K={K} slots, budgets {budgets}, front {len(front)} designs, {res['seconds']['total']:.0f} s")
    for b in budgets:
        print(f"  b={b:3d}  " + "  ".join(f"{m}: {np.mean(tab[(m, b)]):5.1f}%" for m in
                                         ("nsga_gnn", "nsga_linear", "greedy", "random")), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="append")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_toml("engine.toml")
    probe = load_scene(DATA_DIR / "processed" / VAL[0])
    U = np.load(DATA_DIR / "processed" / VAL[0] / "labels.npz")["utci"]
    sg0 = SceneGraph(probe, baseline_utci=U[0], cfg=cfg, device=DEV)
    gnn = load_gnn(DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt", sg0.build(np.zeros(probe.n_slots), 0), DEV)
    cps = calibrate(gnn, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conformal.json").write_text(json.dumps({k: {"alpha": ALPHA, "q_all": v.q, "q_by_hour":
                                                       {int(h): q for h, q in v.q_by_group.items()}}
                                                   for k, v in cps.items()}, indent=2), encoding="utf-8")
    print("conformal q (all hours): gnn %.3f C, linear %.3f C" % (cps["gnn"].q, cps["linear"].q), flush=True)
    for pid in a.patch or TEST:
        run_patch(pid, gnn, cps, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
