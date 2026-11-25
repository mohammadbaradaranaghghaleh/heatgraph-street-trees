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
    (ROOT / "manuscript" / "facts.json").write_text(json.dumps(F, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: F[k] for k in F if k not in ("held_out", "bench", "solweig", "density", "split")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
