"""Second revision round (simulated LUP review R5-R8). No SOLWEIG runs; engine + surrogates only.

    .\\.venv\\Scripts\\python.exe scripts/13_revision_round2.py --part coverage     # R5, ~2 min GPU
    .\\.venv\\Scripts\\python.exe scripts/13_revision_round2.py --part equity       # R7, CPU
    .\\.venv\\Scripts\\python.exe scripts/13_revision_round2.py --part sensitivity  # R6, ~15 min GPU
    .\\.venv\\Scripts\\python.exe scripts/13_revision_round2.py --part seeds        # R8, ~40 min GPU

R5 coverage   Empirical coverage of the hourly split-conformal margin q_h (calibrated on the
              validation patches, alpha = 0.1) on (a) all counterfactual plantings of the test
              patches and (b) the layouts the optimiser returned (NSGA-III GNN front, greedy and
              NSGA-III picks), each against the engine. Covered: |engine - GNN point| <= q_h.
R6 sensitivity  Physics parameters (crown transmissivity, new-tree crown size, street wind factor):
              the Stage 5 greedy, sun x flow heuristic and random layouts are re-simulated with
              the perturbed engine. Objective parameters (gravity beta, UTCI threshold): greedy is
              re-run on the linear surrogate under the new objective and compared with the Stage 5
              greedy layouts (slot Jaccard overlap); all layouts are scored with the new objective.
R7 equity     Per-district per-capita dose without new trees, with greedy and with the
              equity-selected front layout (same selection rule as 08_benchmark.py).
R8 seeds      NSGA-III on the GNN (greedy-seeded, as in Stage 5) for seeds 0-4. Engine-verified
              dose reduction of the picks at each budget, and the equity selection on a front
              subsample of 100 layouts per seed.

Outputs: data/stage5/revision_round2_<part>.json
"""
import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from heatgraph.config import DATA_DIR  # noqa: E402

TEST = ["rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt"]
OUT = DATA_DIR / "stage5"
SEEDS = [0, 1, 2, 3, 4]
N_FRONT_SEEDS = 100


def write(part, obj):
    p = OUT / f"revision_round2_{part}.json"
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"-> {p}", flush=True)


class Ctx:
    """Shared models and calibration (loaded once)."""

    def __init__(self, need_gnn=True):
        from heatgraph.config import load_toml
        self.bm = importlib.import_module("08_benchmark")
        self.rv = importlib.import_module("12_revision_experiments")
        self.cfg = load_toml("engine.toml")
        self.gnn = self.cps = None
        if need_gnn:
            from heatgraph.realscene import load_scene
            from heatgraph.surrogate.graphs import SceneGraph
            from heatgraph.surrogate.io import load_gnn
            bm = self.bm
            probe = load_scene(DATA_DIR / "processed" / bm.VAL[0])
            U = np.load(DATA_DIR / "processed" / bm.VAL[0] / "labels.npz")["utci"]
            sg0 = SceneGraph(probe, baseline_utci=U[0], cfg=self.cfg, device=bm.DEV)
            self.gnn = load_gnn(DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt",
                                sg0.build(np.zeros(probe.n_slots), 0), bm.DEV)
            self.cps = bm.calibrate(self.gnn, self.cfg)
            saved = json.loads((OUT / "conformal.json").read_text())
            assert abs(self.cps["gnn"].q - saved["gnn"]["q_all"]) < 1e-9, "conformal differs from Stage 5"

    def patch(self, pid, cfg=None, scene_graph=True):
        from heatgraph.exposure_setup import districts, floor_area_population, segment_flows
        from heatgraph.realpatch import load_patch
        from heatgraph.realscene import build_scene
        from heatgraph.surrogate.graphs import SceneGraph
        cfg = cfg or self.cfg
        p = load_patch(pid, cfg)
        s = build_scene(p)
        U0 = np.load(DATA_DIR / "processed" / pid / "labels.npz")["utci"][0]
        sg = SceneGraph(s, baseline_utci=U0, cfg=cfg, device=self.bm.DEV) if scene_graph else None
        flow_e = segment_flows(pid, s)
        return dict(p=p, s=s, sg=sg, flow_e=flow_e, groups=districts(s, p.core_mask),
                    pop=floor_area_population(s, p.core_mask), K=s.n_slots, E=s.n_segments)

    def args(self, P, flow_e=None, threshold=26.0):
        H = 36
        flows = np.tile(P["flow_e"] if flow_e is None else flow_e, (H, 1))
        return (flows, P["s"].segment_length_m, np.ones(P["E"]), P["groups"], P["pop"], threshold, "theil")

    def engine(self, P, cfg=None):
        from heatgraph.engine.receptor_engine import ReceptorEngine
        s = P["s"]
        eng = ReceptorEngine(s.building_heights, np.vstack(s.receptors), s.weather, s.lat, s.lon, cfg or self.cfg,
                             device=self.bm.DEV)
        seg = np.concatenate([np.full(len(r), e) for e, r in enumerate(s.receptors)])
        return eng, seg


def stage5(pid):
    r = json.loads((OUT / pid / "results.json").read_text())
    z = np.load(OUT / pid / "designs.npz", allow_pickle=True)
    return r, z


# --------------------------------------------------------------------------- R5
def part_coverage(ctx):
    from heatgraph.optimize.additive import AdditiveEvaluator
    from heatgraph.optimize.allocation import SurrogateEvaluator
    out = {"alpha": 0.1, "patches": {}}
    tot = {"plantings": [0, 0], "plantings_non_baseline": [0, 0], "optimised": [0, 0], "optimised_linear": [0, 0]}
    for pid in TEST:
        P = ctx.patch(pid)
        sg, H = P["sg"], P["sg"].hours
        ev_g = SurrogateEvaluator(ctx.gnn, sg, None, None, None, None, {0: 1})
        ev_l = AdditiveEvaluator(sg, ctx.bm.linear_coef(),None, None, None, None, {0: 1})
        qg = np.array([ctx.cps["gnn"].half_width(np.array([h]))[0] for h in range(H)])
        ql = np.array([ctx.cps["linear"].half_width(np.array([h]))[0] for h in range(H)])
        z = np.load(DATA_DIR / "processed" / pid / "labels.npz")
        X, U = z["plantings"], z["utci"]
        res = np.abs(U - ev_g.predict_utci(X, "point")) <= qg[None, :, None]
        r_, zz = stage5(pid)
        n = len(zz["X"])
        Xo = np.vstack([zz["X"], zz["front"]])
        Uo = zz["engine_utci"]
        names = list(zz["names"])
        reps = {nm: sum(1 for x in r_["rows"] if x["design"] == nm) for nm in names}
        meth = np.concatenate([[nm.rsplit("_", 1)[0]] * reps[nm] for nm in names] + [["front"] * len(zz["front"])])
        keep = np.isin(meth, ["front", "nsga_gnn", "greedy", "nsga_linear"])
        res_o = np.abs(Uo[keep] - ev_g.predict_utci(Xo[keep], "point")) <= qg[None, :, None]
        res_ol = np.abs(Uo[keep] - ev_l.predict_utci(Xo[keep], "point")) <= ql[None, :, None]
        d = {"plantings": float(res.mean()), "plantings_non_baseline": float(res[1:].mean()),
             "optimised": float(res_o.mean()), "optimised_linear": float(res_ol.mean()),
             "optimised_by_hour_min": float(res_o.mean((0, 2)).min()), "n_optimised_layouts": int(keep.sum())}
        for k, a in (("plantings", res), ("plantings_non_baseline", res[1:]), ("optimised", res_o),
                     ("optimised_linear", res_ol)):
            tot[k][0] += int(a.sum()); tot[k][1] += int(a.size)
        out["patches"][pid] = d
        print(pid, {k: round(v, 3) for k, v in d.items()}, flush=True)
    out["pooled"] = {k: v[0] / v[1] for k, v in tot.items()}
    out["q_gnn_by_hour_range"] = [float(min(ctx.cps["gnn"].q_by_group.values())),
                                  float(max(ctx.cps["gnn"].q_by_group.values()))]
    write("coverage", out)


# --------------------------------------------------------------------------- R7
def part_equity(ctx):
    from heatgraph.optimize.allocation import objectives_from_utci
    out = {"districts": "2 x 2 quadrants of the 500 m core, split at the core's centre; each segment is "
                        "assigned to the quadrant of its receptors' centroid", "patches": {}}
    for pid in TEST:
        P = ctx.patch(pid, scene_graph=False)
        r, z = stage5(pid)
        args = ctx.args(P)
        names = list(z["names"])
        idx = {}
        i = 0
        for nm in names:
            m = sum(1 for x in r["rows"] if x["design"] == nm)
            idx[nm] = i
            i += m
        n = i
        Uf = z["engine_utci"][n:]
        fr = r["front_gnn"]
        ft = np.array([x["trees"] for x in fr]); fd = np.array([x["engine_dose"] for x in fr])
        fth = np.array([x["engine_theil"] for x in fr])
        d0 = r["engine_dose_none"]
        _, _, pc0 = objectives_from_utci(z["engine_utci"][[idx["none"]]], *args)
        pc0 = pc0[0]
        rows = []
        for e in r["equity_at_matched_cooling"]:
            b = e["budget"]
            gr = next(x for x in r["rows"] if x["method"] == "greedy" and x["budget"] == b)
            keep = (ft <= b) & (fd <= gr["engine_dose"] + 0.1 * (d0 - gr["engine_dose"]))
            j = int(np.argmin(np.where(keep, fth, np.inf)))
            assert abs(fth[j] - e["front_theil"]) < 1e-12
            _, _, pcg = objectives_from_utci(z["engine_utci"][[idx[f"greedy_{b}"]]], *args)
            _, _, pce = objectives_from_utci(Uf[[j]], *args)
            pcg, pce = pcg[0], pce[0]
            red_g = 100 * (pc0 - pcg) / pc0
            red_e = 100 * (pc0 - pce) / pc0
            order = np.argsort(-pc0)                      # districts by baseline per-capita dose, high first
            rows.append({"budget": b, "theil_change_pct": e["theil_change_pct"],
                         "cooling_kept_pct": e["cooling_kept_pct"],
                         "per_capita_dose_none": pc0.tolist(), "reduction_pct_greedy": red_g.tolist(),
                         "reduction_pct_equity": red_e.tolist(),
                         "highest_dose_district": int(order[0]), "lowest_dose_district": int(order[-1]),
                         "gain_highest_pp": float(red_e[order[0]] - red_g[order[0]]),
                         "gain_lowest_pp": float(red_e[order[-1]] - red_g[order[-1]]),
                         "any_district_worse_than_none": bool((pce > pc0 + 1e-9).any())})
        out["patches"][pid] = {"population_proxy": P["pop"], "rows": rows}
        print(pid, [(x["budget"], round(x["gain_highest_pp"], 2), round(x["gain_lowest_pp"], 2)) for x in rows],
              flush=True)
    write("equity", out)


# --------------------------------------------------------------------------- R6
PHYS = {"transmissivity_0.01": ("shadow", "tree_transmissivity", 0.01),
        "transmissivity_0.10": ("shadow", "tree_transmissivity", 0.10),
        "wind_factor_0.3": ("wind", "urban_reduction_factor", 0.3),
        "wind_factor_0.7": ("wind", "urban_reduction_factor", 0.7),
        "crown_young_r2_h6": ("crown", None, (2, 6.0)),
        "crown_mature_r4_h12": ("crown", None, (4, 12.0))}
OBJ = {"beta_0.001": ("beta", 0.001), "beta_0.004": ("beta", 0.004),
       "threshold_24": ("threshold", 24.0), "threshold_28": ("threshold", 28.0)}


def part_sensitivity(ctx):
    from heatgraph.graph.flows import gravity_flows  # noqa: F401  (imported for the beta scenarios)
    from heatgraph.optimize.additive import AdditiveEvaluator
    from heatgraph.optimize.allocation import greedy_path, objectives_from_utci
    out = {"scenarios": {}, "notes": "physics scenarios re-simulate the baseline-optimised layouts; objective "
                                     "scenarios re-run greedy on the linear surrogate"}
    for pid in TEST:
        t0 = time.time()
        P = ctx.patch(pid)
        r, z = stage5(pid)
        budgets, K = r["budgets"], P["K"]
        idx, i = {}, 0
        for nm in list(z["names"]):
            m = sum(1 for x in r["rows"] if x["design"] == nm)
            idx[nm] = (i, m)
            i += m
        heur = ctx.rv.heuristic_path(P["s"], P["sg"], P["flow_e"], budgets[-1], True)
        D = {"none": z["X"][[idx["none"][0]]]}
        for b in budgets:
            D[f"greedy_{b}"] = z["X"][[idx[f"greedy_{b}"][0]]]
            D[f"heur_{b}"] = heur[b - 1][None]
            a, m = idx[f"random_{b}"]
            D[f"random_{b}"] = z["X"][a:a + m]
        names = list(D)
        X = np.vstack([D[n] for n in names])
        spans, j = {}, 0
        for n in names:
            spans[n] = (j, len(D[n])); j += len(D[n])

        def summarise(U_or_dose, args=None):
            dose = U_or_dose if args is None else objectives_from_utci(U_or_dose, *args)[0]
            d0 = dose[0]
            red = lambda n: 100 * (d0 - dose[spans[n][0]:spans[n][0] + spans[n][1]]) / d0
            return {str(b): {"greedy": float(red(f"greedy_{b}")[0]), "heur": float(red(f"heur_{b}")[0]),
                             "random_mean": float(red(f"random_{b}").mean())} for b in budgets}

        eng, seg = ctx.engine(P)
        _, _, Ubase = ctx.bm.engine_eval(eng, P["s"], seg, X, ctx.args(P))
        out["scenarios"].setdefault("baseline", {})[pid] = summarise(Ubase, ctx.args(P))
        # physics scenarios
        for name, (sec, key, val) in PHYS.items():
            cfg = copy.deepcopy(ctx.cfg)
            s = P["s"]
            old = (s.crown_radius, s.crown_height)
            if sec == "crown":
                s.crown_radius, s.crown_height = val
            else:
                cfg[sec][key] = val
            eng2, _ = ctx.engine(P, cfg)
            _, _, U = ctx.bm.engine_eval(eng2, s, seg, X, ctx.args(P))
            s.crown_radius, s.crown_height = old
            out["scenarios"].setdefault(name, {})[pid] = summarise(U, ctx.args(P))
            print(f"  {pid} {name} done", flush=True)
        # objective scenarios
        coef = ctx.bm.linear_coef()
        g_base = {b: D[f"greedy_{b}"][0] for b in budgets}
        for name, (kind, val) in OBJ.items():
            flow_e, thr = P["flow_e"], 26.0
            if kind == "beta":
                flow_e = flows_with_beta(pid, P["s"], val)
            else:
                thr = val
            args = ctx.args(P, flow_e, thr)
            ev_l = AdditiveEvaluator(P["sg"], coef, *args[:5], conformal=ctx.cps["linear"],
                                     hour_groups=list(range(P["sg"].hours)), threshold_c=thr)
            gp = greedy_path(ev_l, budgets[-1])
            Xg = np.vstack([gp[b - 1] for b in budgets])
            _, _, Ug = ctx.bm.engine_eval(eng, P["s"], seg, Xg, args)
            dose_all = objectives_from_utci(np.concatenate([Ubase, Ug]), *args)[0]
            d0 = dose_all[0]
            summ = summarise(dose_all[:len(X)])
            for k, b in enumerate(budgets):
                new = gp[b - 1].astype(bool); old_ = g_base[b].astype(bool)
                summ[str(b)]["greedy_reoptimised"] = float(100 * (d0 - dose_all[len(X) + k]) / d0)
                summ[str(b)]["jaccard_vs_baseline_greedy"] = float((new & old_).sum() / max((new | old_).sum(), 1))
            out["scenarios"].setdefault(name, {})[pid] = summ
            print(f"  {pid} {name} done", flush=True)
        print(f"{pid}: {time.time() - t0:.0f} s", flush=True)
    write("sensitivity", out)


def flows_with_beta(pid, scene, beta):
    import geopandas as gpd
    import networkx as nx
    from scipy.spatial import cKDTree

    from heatgraph.exposure_setup import WALK_RADIUS_M
    from heatgraph.graph.flows import gravity_flows
    d = DATA_DIR / "raw" / pid
    G = nx.read_graphml(d / "walk_network.graphml")
    for _, _, data in G.edges(data=True):
        data["length"] = float(data["length"])
    ids = list(G.nodes)
    xy = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in ids])
    dest = gpd.read_file(d / "network.gpkg", layer="destinations")
    _, near = cKDTree(xy).query(np.column_stack([dest.geometry.x, dest.geometry.y]))
    attract = {}
    for j in near:
        attract[ids[j]] = attract.get(ids[j], 0.0) + 1.0
    f = gravity_flows(G, {n: 1.0 for n in ids}, attract, radius_m=WALK_RADIUS_M, beta=beta, max_paths=1)
    key = lambda u, v: (u, v) if str(u) <= str(v) else (v, u)
    return np.array([f.get(key(str(u), str(v)), 0.0) for u, v in scene.edges], float)


# --------------------------------------------------------------------------- R8
def part_seeds(ctx):
    from heatgraph.optimize.additive import AdditiveEvaluator
    from heatgraph.optimize.allocation import SurrogateEvaluator, greedy_path, run_nsga3
    out = {"seeds": SEEDS, "n_front_engine": N_FRONT_SEEDS, "patches": {}}
    for pid in TEST:
        P = ctx.patch(pid)
        r, _ = stage5(pid)
        budgets, H = r["budgets"], P["sg"].hours
        args = ctx.args(P)
        ev_g = SurrogateEvaluator(ctx.gnn, P["sg"], *args[:5], conformal=ctx.cps["gnn"], hour_groups=list(range(H)))
        ev_l = AdditiveEvaluator(P["sg"], ctx.bm.linear_coef(), *args[:5], conformal=ctx.cps["linear"],
                                 hour_groups=list(range(H)))
        greedy = greedy_path(ev_l, budgets[-1])
        eng, seg = ctx.engine(P)
        d0 = r["engine_dose_none"]
        gr = {b: next(x for x in r["rows"] if x["method"] == "greedy" and x["budget"] == b) for b in budgets}
        per_seed = []
        for sd in SEEDS:
            t = time.time()
            ng = run_nsga3(ev_g, budgets[-1], pop_size=100, n_gen=150, seed=sd, greedy=greedy)
            picks = np.vstack([ctx.bm.pick(ng["archive_X"], ng["archive_F"], b) for b in budgets])
            front = ng["X"][ng["X"].sum(1) <= budgets[-1]]
            order = np.argsort(front.sum(1), kind="stable")
            front = front[order[np.linspace(0, len(front) - 1, min(N_FRONT_SEEDS, len(front))).round().astype(int)]]
            dose, theil, _ = ctx.bm.engine_eval(eng, P["s"], seg, np.vstack([picks, front]), args)
            fdose, fth, ft = dose[len(budgets):], theil[len(budgets):], front.sum(1)
            row = {"seed": sd, "dose_reduction_pct": {}, "theil_change_pct": {}}
            for k, b in enumerate(budgets):
                row["dose_reduction_pct"][str(b)] = float(100 * (d0 - dose[k]) / d0)
                keep = (ft <= b) & (fdose <= gr[b]["engine_dose"] + 0.1 * (d0 - gr[b]["engine_dose"]))
                row["theil_change_pct"][str(b)] = (float(100 * (fth[keep].min() - gr[b]["engine_theil"]) /
                                                         gr[b]["engine_theil"]) if keep.any() else None)
            per_seed.append(row)
            print(f"  {pid} seed {sd}: {time.time() - t:.0f} s", flush=True)
        out["patches"][pid] = {"budgets": budgets, "greedy_pct": {str(b): gr[b]["engine_dose_reduction_pct"]
                                                                   for b in budgets}, "runs": per_seed}
    write("seeds", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", choices=["coverage", "equity", "sensitivity", "seeds"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parts = a.part or ["coverage", "equity", "sensitivity", "seeds"]
    ctx = Ctx(need_gnn=any(p != "equity" for p in parts))
    for p in parts:
        {"coverage": part_coverage, "equity": part_equity, "sensitivity": part_sensitivity,
         "seeds": part_seeds}[p](ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
