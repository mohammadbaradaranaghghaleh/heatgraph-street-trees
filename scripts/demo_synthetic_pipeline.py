"""Synthetic end-to-end demo (CPU only): engine labels -> GNN -> conformal -> NSGA-III -> re-simulation.

    .\\.venv\\Scripts\\python.exe scripts/demo_synthetic_pipeline.py
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.optimize.allocation import (SurrogateEvaluator, compare_fronts, greedy_path,  # noqa: E402
                                           is_nondominated, run_nsga3)
from heatgraph.surrogate.graphs import SceneGraph  # noqa: E402
from heatgraph.surrogate.model import HeatGraphGNN  # noqa: E402
from heatgraph.surrogate.train import evaluate, fit, predict  # noqa: E402
from heatgraph.synthetic import generate_labels, random_plantings, toy_scene  # noqa: E402
from heatgraph.uq.conformal import SplitConformal  # noqa: E402
from heatgraph.validate.pareto_gap import surrogate_optimality_gap  # noqa: E402
from test_pipeline_e2e import segment_flows  # noqa: E402


def main(slot_conv="edgemlp"):
    torch.manual_seed(0)
    s = toy_scene()
    X = random_plantings(s, 70, 0.35, 1)
    Y = generate_labels(s, X)
    sg = SceneGraph(s, baseline_utci=Y[0])
    tr, va, ca, te = [sg.dataset(X[a:b], Y[a:b]) for a, b in ((0, 40), (40, 50), (50, 60), (60, 70))]
    m = HeatGraphGNN(hidden=48, slot_conv=slot_conv)
    fit(m, tr, va, epochs=80)
    r = evaluate(m, te)
    dt = (Y[60:] - Y[0]).ravel()
    dp = (r["pred"].reshape(10, sg.hours, -1) - Y[0]).ravel()
    yc = np.concatenate([d["segment"].y.numpy() for d in ca])
    cp = SplitConformal(0.1).calibrate(yc, predict(m, ca))
    lo, hi = cp.interval(r["pred"])
    print(f"[{slot_conv}] test MAE {r['mae']:.3f} C | change corr {np.corrcoef(dt, dp)[0, 1]:.3f} | "
          f"change MAE {np.mean(np.abs(dt - dp)):.3f} vs zero-change {np.mean(np.abs(dt)):.3f} | "
          f"conformal half-width {cp.q:.3f} C, test coverage {SplitConformal.coverage(r['true'], lo, hi):.3f}")
    if slot_conv != "edgemlp":
        return
    fl, le = segment_flows(s)
    ev = SurrogateEvaluator(m, sg, fl, le, np.ones(s.n_segments), s.group, s.population, conformal=cp)
    gpath = greedy_path(ev, 8)
    Fg = ev(gpath)
    res = run_nsga3(ev, budget=8, pop_size=30, n_gen=15, seed=0, greedy=gpath)
    plain = run_nsga3(ev, budget=8, pop_size=30, n_gen=15, seed=0, seed_with_greedy=False)
    F, Xp = res["F"], res["X"]
    zero = ev(np.zeros((1, s.n_slots), int))[0]
    hv = compare_fronts({"seeded": F, "unseeded": plain["F"], "greedy": Fg[is_nondominated(Fg)]})
    print(f"Pareto designs {len(Xp)} | tree counts {sorted(set(Xp.sum(1).tolist()))} | min dose: seeded "
          f"{F[:, 0].min():.1f}, unseeded {plain['F'][:, 0].min():.1f}, greedy {Fg[:, 0].min():.1f}, none {zero[0]:.1f}")
    print(f"hypervolume (shared ref point): seeded {hv['seeded']:.4g} | unseeded {hv['unseeded']:.4g} | "
          f"greedy path {hv['greedy']:.4g}")
    pick = Xp[np.argsort(F[:, 0])[:4]]
    gap = surrogate_optimality_gap(ev, pick, generate_labels(s, pick))
    print("re-simulated rel. dose gap", np.round(gap["rel_gap"], 3), "| upper bound covers engine", gap["upper_covers"])


if __name__ == "__main__":
    main("edgemlp")
    main("gatv2")
