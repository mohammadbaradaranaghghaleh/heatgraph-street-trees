"""Canonical numeric facts for the manuscript, read ONLY from saved result files.

    .\\.venv\\Scripts\\python.exe manuscript/build_facts.py   -> manuscript/facts.json

Every number quoted in main.tex must come from this file; manuscript/check_numbers.py audits it.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    import numpy as np
    import tomllib

    F = {}
    # ---- patches and data
    rows = list(csv.DictReader(open(D / "raw" / "ingest_summary.csv", encoding="utf-8")))
    cfg = tomllib.load(open(ROOT / "configs" / "patches.toml", "rb"))
    nl = [p for p in cfg["patch"]]
    F["n_patches"] = len(rows)
    F["n_nl_patches"] = len(nl)
    F["n_nl_cities"] = len({p["city"] for p in nl})
    F["nl_cities"] = sorted({p["city"] for p in nl})
    F["n_cities_total"] = F["n_nl_cities"] + 1
    frozen = j(D / "raw" / "FROZEN.json")
    F["n_frozen_files"] = sum(v["n_files"] for v in frozen["patches"].values())
    F["freiburg_height_missing_pct"] = float(next(r for r in rows if r["patch"].startswith("xfer_"))["height_missing_pct"])
    F["raw_mb"] = round(sum(float(r["mb"]) for r in rows))
    # ---- engine validation
    c = j(D / "validation" / "calibration.json")
    F["calibration"] = {"params": c["calibration"]["params"], "before": c["calibration"]["before"],
                        "after": c["calibration"]["after"]}
    F["held_out"] = {k: {"default": v["default"], "calibrated": v["calibrated"],
                         "daytime": {s: v["daytime_calibrated"][s]["bias"] for s in ("sunlit", "shaded", "under_canopy")}}
                     for k, v in c["held_out"].items()}
    mae = [v["calibrated"]["mae_all"] for v in c["held_out"].values()]
    bias = [v["calibrated"]["bias_all"] for v in c["held_out"].values()]
    F["held_out_mae_range"] = [min(mae), max(mae)]
    F["held_out_bias_range"] = [min(bias), max(bias)]
    sh = [v["daytime_calibrated"]["shaded"]["bias"] for v in c["held_out"].values()]
    F["shaded_bias_range"] = [min(sh), max(sh)]
    # ---- dataset
    metas = [j(p) for p in sorted((D / "processed").glob("*/meta.json"))]
    F["n_plantings_total"] = sum(m["n_plantings"] for m in metas)
    F["n_plantings_per_patch"] = metas[0]["n_plantings"]
    F["hours"] = metas[0]["hours"]
    F["n_graphs_total"] = sum(m["n_plantings"] * m["hours"] for m in metas)
    F["n_segments_total"] = sum(m["n_segments"] for m in metas)
    F["n_receptors_total"] = sum(m["n_receptors"] for m in metas)
    F["slots_range"] = [min(m["n_slots"] for m in metas), max(m["n_slots"] for m in metas)]
    F["mean_tree_effect_range"] = [min(m["mean_tree_effect_c"] for m in metas), max(m["mean_tree_effect_c"] for m in metas)]
    F["max_cooling_range"] = [min(m["max_tree_cooling_c"] for m in metas), max(m["max_tree_cooling_c"] for m in metas)]
    F["seconds_per_patch_range"] = [min(m["seconds"] for m in metas), max(m["seconds"] for m in metas)]
    # ---- surrogates
    M = D / "models"
    for name, f in [("gatv2", "metrics_gatv2.json"), ("v1_within", "metrics_edgemlp.json"),
                    ("v1_patch", "metrics_edgemlp_v1_patchval.json"), ("v2", "metrics_edgemlp_v2_patchval.json"),
                    ("linear", "metrics_linear_physics.json")]:
        r = j(M / f)
        F[f"sur_{name}"] = {k: r[k] for k in ("val", "test_city", "transfer") if k in r}
        if "history" in r:
            F[f"sur_{name}"]["best_epoch"] = r["history"].get("best_epoch")
            F[f"sur_{name}"]["train_seconds"] = r.get("train_seconds")
            F[f"sur_{name}"]["n_train_graphs"] = r["split"]["n_train_graphs"]
    F["split"] = j(M / "metrics_edgemlp_v2_patchval.json")["split"]
    F["density"] = j(M / "density_analysis.json")["by_bin"]
    con = j(D / "stage5" / "conformal.json")
    F["conformal"] = {k: {"q_all": v["q_all"], "alpha": v["alpha"]} for k, v in con.items()}
    # ---- benchmark
    B = {}
    for p in ("rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt"):
        r = j(D / "stage5" / p / "results.json")
        tab = {}
        for x in r["rows"]:
            tab.setdefault((x["method"], x["budget"]), []).append(x["engine_dose_reduction_pct"])
        B[p] = {"budgets": r["budgets"], "n_slots": r["n_slots"],
                "mean": {f"{m}_{b}": float(np.mean(tab[(m, b)])) for (m, b) in tab},
                "equity": r["equity_at_matched_cooling"], "evals": r["evals"], "seconds": r["seconds"]}
    F["bench"] = B
    ratio = []
    for p, b in B.items():
        for bb in b["budgets"]:
            ratio.append(b["mean"][f"greedy_{bb}"] / b["mean"][f"random_{bb}"])
    F["opt_vs_random_ratio_range"] = [min(ratio), max(ratio)]
    top = [(b["mean"][f"greedy_{b['budgets'][-1]}"], b["mean"][f"random_{b['budgets'][-1]}"]) for b in B.values()]
    F["top_budget_greedy_range"] = [min(t[0] for t in top), max(t[0] for t in top)]
    F["top_budget_random_range"] = [min(t[1] for t in top), max(t[1] for t in top)]
    diff = [b["mean"][f"nsga_gnn_{bb}"] - b["mean"][f"greedy_{bb}"] for b in B.values() for bb in b["budgets"]]
    F["nsga_minus_greedy_pp_range"] = [min(diff), max(diff)]
    eq = [(e["theil_change_pct"], e["budget"], p) for p, b in B.items() for e in b["equity"]]
    F["equity_best"] = min(eq)
    # ---- SOLWEIG verification
    V = j(D / "stage5" / "rot_centrum" / "solweig_verification.json")
    F["solweig"] = V["designs"]
    under_e = [1 - d["engine"]["dose_reduction_pct"] / d["solweig"]["dose_reduction_pct"]
               for n, d in V["designs"].items() if n != "none"]
    under_g = [1 - d["gnn"]["dose_reduction_pct"] / d["solweig"]["dose_reduction_pct"]
               for n, d in V["designs"].items() if n != "none"]
    F["underestimate_engine_pct_range"] = [100 * min(under_e), 100 * max(under_e)]
    F["underestimate_gnn_pct_range"] = [100 * min(under_g), 100 * max(under_g)]
    a = [d["agreement"] for n, d in V["designs"].items() if n != "none"]
    F["corr_gnn_solweig_range"] = [min(x["corr_gnn_vs_solweig"] for x in a), max(x["corr_gnn_vs_solweig"] for x in a)]
    F["corr_engine_solweig_range"] = [min(x["corr_engine_vs_solweig"] for x in a), max(x["corr_engine_vs_solweig"] for x in a)]
    # ---- revision R4: SOLWEIG comparison on a common mask (scripts/12_revision_experiments.py --part verify)
    R4 = j(D / "stage5" / "revision_r4_verification.json")
    F["r4"] = R4["designs"]
    rng = lambda xs: [min(xs), max(xs)]
    r4 = list(R4["designs"].values())
    F["r4_mean_under_engine_pct_range"] = rng([100 * (1 - d["engine"]["mean_dUTCI_common_mask"] /
                                                       d["solweig"]["mean_dUTCI_common_mask"]) for d in r4])
    F["r4_mean_under_gnn_pct_range"] = rng([100 * (1 - d["gnn"]["mean_dUTCI_common_mask"] /
                                                    d["solweig"]["mean_dUTCI_common_mask"]) for d in r4])
    F["r4_bias_strong_engine_range"] = rng([d["engine"]["bias_strong_cooling"] for d in r4])
    F["r4_bias_weak_engine_range"] = rng([d["engine"]["bias_weak_cooling"] for d in r4])
    F["r4_bias_strong_gnn_range"] = rng([d["gnn"]["bias_strong_cooling"] for d in r4])
    F["r4_spread_engine_range"] = rng([d["engine"]["spread_ratio_vs_solweig"] for d in r4])
    F["r4_spread_gnn_range"] = rng([d["gnn"]["spread_ratio_vs_solweig"] for d in r4])
    # ---- revision R2/R3: practitioner heuristics and unseeded NSGA-III (scripts/12_revision_experiments.py --part bench)
    RV = {p: j(D / "stage5" / p / "revision_r2r3.json") for p in B
          if (D / "stage5" / p / "revision_r2r3.json").exists()}
    if RV:
        share = {m: [r["share_of_greedy_pct"] for v in RV.values() for r in v["rows"] if r["method"] == m]
                 for m in ("heur_sunflow", "heur_sun", "nsga_unseeded")}
        F["rev_share_of_greedy_pct_range"] = {m: rng(v) for m, v in share.items()}
        F["rev_heur_vs_random_ratio_range"] = {
            m: rng([r["engine_dose_reduction_pct"] / r["random_mean_pct"] for v in RV.values() for r in v["rows"]
                    if r["method"] == m]) for m in ("heur_sunflow", "heur_sun")}
        F["rev_unseeded_minus_greedy_pp_range"] = rng([r["engine_dose_reduction_pct"] - r["greedy_pct"]
                                                       for v in RV.values() for r in v["rows"]
                                                       if r["method"] == "nsga_unseeded"])
        F["rev_hv"] = {p: {k: v["hypervolume_engine_normalised"][k] for k in ("seeded", "unseeded", "greedy_budgets")}
                       for p, v in RV.items()}
        F["rev_hv_unseeded_vs_seeded_pct_range"] = rng([100 * (h["unseeded"] / h["seeded"] - 1)
                                                        for h in F["rev_hv"].values()])
        F["rev_bench"] = {p: {f"{r['method']}_{r['budget']}": r["engine_dose_reduction_pct"] for r in v["rows"]}
                          for p, v in RV.items()}
    # ---- revision round 2 (scripts/13_revision_round2.py)
    p = D / "stage5" / "revision_round2_coverage.json"
    if p.exists():
        C = j(p)
        F["r5_cov_pooled_plantings"] = C["pooled"]["plantings"]
        F["r5_cov_plantings_range"] = rng([v["plantings"] for v in C["patches"].values()])
        F["r5_cov_optimised_range"] = rng([v["optimised"] for v in C["patches"].values()])
        F["r5_cov_optimised_hour_min"] = min(v["optimised_by_hour_min"] for v in C["patches"].values())
        F["r5_q_by_hour_range"] = C["q_gnn_by_hour_range"]
    p = D / "stage5" / "revision_round2_equity.json"
    if p.exists():
        Q = [r for v in j(p)["patches"].values() for r in v["rows"]]
        hi = [r["gain_highest_pp"] for r in Q]
        lo = [r["gain_lowest_pp"] for r in Q]
        F["r7_n_cases"] = len(Q)
        F["r7_n_lowest_less"] = sum(x < -0.005 for x in lo)
        F["r7_n_highest_more"] = sum(x > 0.005 for x in hi)
        F["r7_lowest_loss_max_pp"] = -min(lo)
        F["r7_highest_gain_max_pp"] = max(hi)
        F["r7_highest_gain_median_pp"] = float(np.median(hi))
        F["r7_any_worse_than_none"] = any(r["any_district_worse_than_none"] for r in Q)
    p = D / "stage5" / "revision_round2_seeds.json"
    if p.exists():
        S = j(p)
        sd_dose, gap, th_sd, th_mean = [], [], [], []
        for v in S["patches"].values():
            for b in v["budgets"]:
                x = [r["dose_reduction_pct"][str(b)] for r in v["runs"]]
                sd_dose.append(float(np.std(x, ddof=1)))
                gap += [xi - v["greedy_pct"][str(b)] for xi in x]
                t = [r["theil_change_pct"][str(b)] for r in v["runs"] if r["theil_change_pct"][str(b)] is not None]
                if len(t) > 1:
                    th_sd.append(float(np.std(t, ddof=1))); th_mean.append(float(np.mean(t)))
        F["r8_n_seeds"] = len(S["seeds"])
        F["r8_dose_sd_max_pp"] = max(sd_dose)
        F["r8_gap_vs_greedy_range_pp"] = rng(gap)
        F["r8_theil_sd_range"] = rng(th_sd)
        F["r8_theil_mean_best"] = min(th_mean)
        allt = [r["theil_change_pct"][str(b)] for v in S["patches"].values() for b in v["budgets"] for r in v["runs"]]
        F["r8_theil_n_total"] = len(allt)
        F["r8_theil_n_improve"] = sum(t is not None and t < -0.05 for t in allt)
        F["r8_theil_n_worse"] = sum(t is not None and t > 0.05 for t in allt)
        F["r8_theil_median"] = float(np.median([t for t in allt if t is not None]))
    p = D / "stage5" / "revision_round2_sensitivity.json"
    if p.exists():
        SN = j(p)["scenarios"]
        sens = {}
        for name, pats in SN.items():
            ratio, share, red40, jac, reopt = [], [], [], [], []
            for v in pats.values():
                bs = sorted(v, key=int)
                for b in bs:
                    x = v[b]
                    ratio.append(x["greedy"] / x["random_mean"])
                    share.append(100 * x["heur"] / x["greedy"])
                    if "jaccard_vs_baseline_greedy" in x:
                        jac.append(x["jaccard_vs_baseline_greedy"])
                        reopt.append(x["greedy_reoptimised"] - x["greedy"])
                red40.append(v[bs[-1]]["greedy"])
            sens[name] = {"ratio_range": rng(ratio), "heur_share_range": rng(share), "greedy_top_range": rng(red40)}
            if jac:
                sens[name]["jaccard_range"] = rng(jac)
                sens[name]["reopt_gain_pp_range"] = rng(reopt)
        F["r6"] = sens
        F["r6_ratio_min_all"] = min(v["ratio_range"][0] for v in sens.values())
        F["r6_heur_share_max_all"] = max(v["heur_share_range"][1] for v in sens.values())
    (ROOT / "manuscript" / "facts.json").write_text(json.dumps(F, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: F[k] for k in F if k not in ("held_out", "bench", "solweig", "density", "split")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
