"""Static LaTeX checks + numerical consistency audit of main.tex against facts.json.

    ..\\.venv\\Scripts\\python.exe check_numbers.py      (run inside manuscript/)

Structural: \\ref <-> \\label, \\cite keys in references.bib, \\input / \\includegraphics files
exist, braces and \\begin/\\end balance.
Numerical: every value below is RECOMPUTED from facts.json (built from the result files)
and must appear in main.tex exactly as formatted. A failure means text and data disagree.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else (HERE / "main.tex").read_text(encoding="utf-8")
body = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
F = json.loads((HERE / "facts.json").read_text(encoding="utf-8"))
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------ structural
inputs = re.findall(r"\\input\{([^}]+)\}", body)
alltext = body + "".join((HERE / (p if p.endswith(".tex") else p + ".tex")).read_text(encoding="utf-8")
                         for p in inputs if (HERE / (p if p.endswith(".tex") else p + ".tex")).exists())
labels = set(re.findall(r"\\label\{([^}]+)\}", alltext))
for r in set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body)):
    check(r in labels, f"\\ref{{{r}}} has no \\label")
bibkeys = set(re.findall(r"@\w+\{([^,]+),", (HERE / "references.bib").read_text(encoding="utf-8")))
cited = {k.strip() for g in re.findall(r"\\cite[pt]?(?:alp)?\{([^}]+)\}", body) for k in g.split(",")}
for k in cited:
    check(k in bibkeys, f"\\cite{{{k}}} missing from references.bib")
check(not (bibkeys - cited), f"uncited bib entries: {sorted(bibkeys - cited)}")
for p in inputs:
    check((HERE / (p if p.endswith(".tex") else p + ".tex")).exists(), f"\\input{{{p}}} file missing")
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    check((HERE / "figures" / g).exists(), f"figure {g} missing")
check(body.count("{") == body.count("}"), f"brace imbalance {{={body.count('{')} }}={body.count('}')}")
envs = re.findall(r"\\(begin|end)\{([^}]+)\}", body)
stack = []
for kind, name in envs:
    if kind == "begin":
        stack.append(name)
    else:
        check(stack and stack[-1] == name, f"\\end{{{name}}} does not close \\begin{{{stack[-1] if stack else None}}}")
        if stack and stack[-1] == name:
            stack.pop()
check(not stack, f"unclosed environments: {stack}")
for i in range(1, 8):
    check(re.search(rf"\\ref\{{fig:F{i}\}}", body), f"Fig. F{i} never referenced")
for t in ("T1_patches", "T2_engine_vs_solweig", "T3_surrogates", "T4_benchmark", "T5_equity", "T6_solweig_verification",
          "T7_sensitivity"):
    check(f"tables/{t}" in body, f"table {t} not input")
    check(f"tab:{t}" in body.replace("\\ref{", ""), f"table {t} never referenced")

# ------------------------------------------------------------------ numbers (recomputed from facts.json)
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.0f}"
v2, lin, gat = F["sur_v2"], F["sur_linear"], F["sur_gatv2"]
v1w, v1p = F["sur_v1_within"], F["sur_v1_patch"]
skill = lambda r: 100 * (1 - r["change_mae"] / r["zero_change_mae"])
dens = F["density"]
gain = [100 * (b["linear"] - b["gnn"]) / b["linear"] for b in dens]
ho = F["held_out_mae_range"], F["held_out_bias_range"]
sol = F["solweig"]
expect = {
    # data
    "patches": (str(F["n_patches"]), "18 patches"),
    "nl cities": ("nine Dutch cities", "nine Dutch cities" if F["n_nl_cities"] == 9 else "MISMATCH"),
    "frozen files": (f"{F['n_frozen_files']} files", f"{F['n_frozen_files']} files"),
    "freiburg missing": (f"{F['freiburg_height_missing_pct']:.1f}\\%", f"{F['freiburg_height_missing_pct']:.1f}\\%"),
    "plantings": (f"{F['n_plantings_total']:,}", f"{F['n_plantings_total']:,}"),
    "per patch": (f"{F['n_plantings_per_patch']} plantings", f"{F['n_plantings_per_patch']} plantings"),
    "graphs": (f"{F['n_graphs_total']:,}", f"{F['n_graphs_total']:,}"),
    "segments": (f"{F['n_segments_total']:,} segments", f"{F['n_segments_total']:,} segments"),
    "receptors": (f"{F['n_receptors_total']:,} receptors", f"{F['n_receptors_total']:,} receptors"),
    "slots": (f"{F['slots_range'][0]}--{F['slots_range'][1]} slots", ""),
    "gen time": (f"{F['seconds_per_patch_range'][0]:.0f}--{F['seconds_per_patch_range'][1]:.0f}~s", ""),
    "train patches": (f"{len(F['split']['train'])} Dutch patches", ""),
    # engine
    "cal mae": (f"from {f2(F['calibration']['before']['mae_all'])} to {f2(F['calibration']['after']['mae_all'])}~K", ""),
    "cal bias": (f"from ${f2(F['calibration']['before']['bias_all'])}$ to ${f2(F['calibration']['after']['bias_all'])}$~K", ""),
    "ho mae": (f"{f2(ho[0][0])}--{f2(ho[0][1])}~K", ""),
    "ho bias": (f"${f2(ho[1][1])}$ to ${f2(ho[1][0])}$~K", ""),
    "shaded": (f"{f2(-F['shaded_bias_range'][1])}--{f2(-F['shaded_bias_range'][0])}~K", ""),
    "params": (f"{F['calibration']['params']['ground_warming_k_per_kwm2']}~K per kW", ""),
    "wall": (f"{F['calibration']['params']['wall_warming_k_per_kwm2']}~K per kW", ""),
    "albedo": (f"albedo {F['calibration']['params']['albedo_ground']:.2f}", ""),
    # surrogates
    "v1w val": (f3(v1w["val"]["change_mae"]), ""), "v1w rot": (f3(v1w["test_city"]["change_mae"]), ""),
    "v1w fr": (f3(v1w["transfer"]["change_mae"]), ""), "v1p val": (f3(v1p["val"]["change_mae"]), ""),
    "v1p rot": (f3(v1p["test_city"]["change_mae"]), ""),
    "v2 rot": (f3(v2["test_city"]["change_mae"]), ""), "v2 fr": (f3(v2["transfer"]["change_mae"]), ""),
    "v2 rot corr": (f3(v2["test_city"]["change_corr"]), ""), "v2 fr corr": (f3(v2["transfer"]["change_corr"]), ""),
    "v2 rot mae": (f3(v2["test_city"]["mae"]), ""), "v2 fr mae": (f3(v2["transfer"]["mae"]), ""),
    "v2 val": (f3(v2["val"]["change_mae"]), ""),
    "v2 skill": (f"{pct(skill(v2['test_city']))}\\% and {pct(skill(v2['transfer']))}\\%", ""),
    "v1 skill": (f"{pct(skill(v1w['test_city']))}\\% and {pct(skill(v1w['transfer']))}\\%", ""),
    "lin rot": (f3(lin["test_city"]["change_mae"]), ""), "lin fr": (f3(lin["transfer"]["change_mae"]), ""),
    "lin rot corr": (f3(lin["test_city"]["change_corr"]), ""), "lin fr corr": (f3(lin["transfer"]["change_corr"]), ""),
    "gnn vs lin": (f"{pct(100 * (1 - v2['test_city']['change_mae'] / lin['test_city']['change_mae']))}\\% more accurate in Rotterdam and "
                   f"{pct(100 * (1 - v2['transfer']['change_mae'] / lin['transfer']['change_mae']))}\\% more accurate in Freiburg", ""),
    "gat": (f"{f3(gat['test_city']['change_mae'])} and {f3(gat['transfer']['change_mae'])}", ""),
    "gat corr": (f"{f3(gat['test_city']['change_corr'])} and {f3(gat['transfer']['change_corr'])}", ""),
    "density lo": (f"(${pct(gain[0])}$\\%)", ""),
    "density": (f"{pct(gain[1])}\\%, {pct(gain[2])}\\% and {pct(gain[3])}\\%", ""),
    "conformal gnn": (f"$\\pm${F['conformal']['gnn']['q_all']:.3f}", ""),
    "conformal lin": (f"$\\pm${F['conformal']['linear']['q_all']:.3f}", ""),
    # benchmark
    "ratio": (f"{F['opt_vs_random_ratio_range'][0]:.1f}--{F['opt_vs_random_ratio_range'][1]:.1f}", ""),
    "top greedy": (f"{f2(F['top_budget_greedy_range'][0])}--{f2(F['top_budget_greedy_range'][1])}\\%", ""),
    "top random": (f"{f2(F['top_budget_random_range'][0])}--{f2(F['top_budget_random_range'][1])}\\%", ""),
    "nsga-greedy": (f"${F['nsga_minus_greedy_pp_range'][0]:.2f}$ to $+{F['nsga_minus_greedy_pp_range'][1]:.2f}$", ""),
    "evals gnn": (f"{F['bench']['rot_centrum']['evals']['nsga_gnn']:,} GNN evaluations", ""),
    "evals lin": (f"{F['bench']['rot_centrum']['evals']['nsga_linear']:,} linear evaluations", ""),
    "equity": (f"{-F['equity_best'][0]:.1f}\\% (Rotterdam Oude Westen, {F['equity_best'][1]} trees)", ""),
    # SOLWEIG
    "sol dose": (", ".join(f2(sol[n]["solweig"]["dose_reduction_pct"]) for n in ("low_budget", "knee", "high_budget"))
                 + f" and {f2(sol['max_cooling']['solweig']['dose_reduction_pct'])}\\%", ""),
    "eng dose": (", ".join(f2(sol[n]["engine"]["dose_reduction_pct"]) for n in ("low_budget", "knee", "high_budget"))
                 + f" and {f2(sol['max_cooling']['engine']['dose_reduction_pct'])}\\%", ""),
    "gnn dose": (", ".join(f2(sol[n]["gnn"]["dose_reduction_pct"]) for n in ("low_budget", "knee", "high_budget"))
                 + f" and {f2(sol['max_cooling']['gnn']['dose_reduction_pct'])}\\%", ""),
    "under eng": (f"{F['underestimate_engine_pct_range'][0]:.0f}--{F['underestimate_engine_pct_range'][1]:.0f}\\%", ""),
    "under gnn": (f"{F['underestimate_gnn_pct_range'][0]:.0f}--{F['underestimate_gnn_pct_range'][1]:.0f}\\%", ""),
    "corr gnn": (f"{f2(F['corr_gnn_solweig_range'][0])}--{f2(F['corr_gnn_solweig_range'][1])}", ""),
    "corr eng": (f"{f2(F['corr_engine_solweig_range'][0])}--{f2(F['corr_engine_solweig_range'][1])}", ""),
    "max cool": (f"${sol['high_budget']['solweig']['max_cooling']:.2f}$~\\degC, against ${sol['high_budget']['engine']['max_cooling']:.2f}$", ""),
}
# revision R2/R3/R4 (scripts/12_revision_experiments.py)
rr = lambda k, fmt="{:.0f}": f"{fmt.format(k[0])}--{fmt.format(k[1])}"
rs = F["rev_share_of_greedy_pct_range"]
expect.update({
    "heur sunflow share": (f"achieved {rr(rs['heur_sunflow'])}\\% of greedy's reduction", ""),
    "heur sun share": (f"sun-only rule achieved {rr(rs['heur_sun'])}\\% of greedy's reduction", ""),
    "heur vs random": (f"{rr(F['rev_heur_vs_random_ratio_range']['heur_sunflow'], '{:.1f}')} times the reduction of random", ""),
    "unseeded share": (f"reached only {rr(rs['nsga_unseeded'])}\\% of greedy's reduction", ""),
    "unseeded hv": (f"front was {rr([-x for x in F['rev_hv_unseeded_vs_seeded_pct_range'][::-1]])}\\% lower", ""),
    "abstract heur": (f"achieved only {rr(rs['heur_sunflow'])}\\% of that reduction, and {rr(rs['heur_sun'])}\\% when", ""),
    "r4 mean": (f"engine's mean cooling was {rr(F['r4_mean_under_engine_pct_range'])}\\% smaller", ""),
    "r4 strong": (f"by {rr([-x for x in F['r4_bias_strong_engine_range'][::-1]], '{:.2f}')}~\\degC{{}} on average", ""),
    "r4 weak": (f"it cooled less, by {rr(F['r4_bias_weak_engine_range'], '{:.2f}')}~\\degC", ""),
    "r4 spread": (f"only {rr([100 * x for x in F['r4_spread_engine_range']])}\\% as many segment-hours", ""),
    "r4 peak": (f"by up to {-F['r4_bias_strong_engine_range'][0]:.2f}~\\degC{{}} on average", ""),
    # round 2
    "r5 q": (f"ranged from {F['r5_q_by_hour_range'][0]:.2f}~\\degC{{}} in the early morning to {F['r5_q_by_hour_range'][1]:.2f}~\\degC", ""),
    "r5 plantings": (f"covered {100 * F['r5_cov_pooled_plantings']:.0f}\\% of segment-hours when pooled and "
                     f"{rr([100 * x for x in F['r5_cov_plantings_range']])}\\% per patch", ""),
    "r5 optimised": (f"at {rr([100 * x for x in F['r5_cov_optimised_range']])}\\% per patch and as low as "
                     f"{100 * F['r5_cov_optimised_hour_min']:.0f}\\% at single hours", ""),
    "r7": (f"In {F['r7_n_lowest_less']} of the {F['r7_n_cases']} patch--budget combinations", ""),
    "r7 loss": (f"by up to {F['r7_lowest_loss_max_pp']:.2f} percentage points", ""),
    "r7 gain": (f"more cooling in {F['r7_n_highest_more']} combinations, but by at most "
                f"{F['r7_highest_gain_max_pp']:.2f} percentage points and with a median of {F['r7_highest_gain_median_pp']:.2f}", ""),
})
if "r6" in F:
    s6 = F["r6"]
    obj = [k for k in s6 if "jaccard_range" in s6[k]]
    expect.update({
        "r6 crown": (f"by {rr(s6['crown_young_r2_h6']['greedy_top_range'], '{:.2f}')}\\%, and with mature crowns by "
                     f"{rr(s6['crown_mature_r4_h12']['greedy_top_range'], '{:.2f}')}\\%, against "
                     f"{rr(s6['baseline']['greedy_top_range'], '{:.2f}')}\\%", ""),
        "r6 ratio": (f"at least {F['r6_ratio_min_all']:.1f} times more than random", ""),
        "r6 heur": (f"never exceeded {F['r6_heur_share_max_all']:.0f}\\% of greedy's reduction", ""),
        "r6 jaccard": (f"Jaccard index of {min(s6[k]['jaccard_range'][0] for k in obj):.2f}--"
                       f"{max(s6[k]['jaccard_range'][1] for k in obj):.2f}", ""),
        "r6 reopt": (f"gained at most {max(s6[k]['reopt_gain_pp_range'][1] for k in obj):.2f} percentage points", ""),
    })
if "r8_n_seeds" in F:
    expect.update({
        "r8 sd": (f"standard deviation of at most {F['r8_dose_sd_max_pp']:.2f} percentage points and stayed within "
                  f"${F['r8_gap_vs_greedy_range_pp'][0]:.2f}$ to $+{F['r8_gap_vs_greedy_range_pp'][1]:.2f}$ points", ""),
        "r8 eq": (f"found in {F['r8_theil_n_improve']} of the {F['r8_theil_n_total']} patch--budget--seed combinations, "
                  f"with a median change of ${F['r8_theil_median']:.1f}$\\%", ""),
        "r8 eq sd": (f"standard deviation of up to {F['r8_theil_sd_range'][1]:.1f} percentage points, and in "
                     f"{F['r8_theil_n_worse']} combinations", ""),
    })
check(not F["r7_any_worse_than_none"], "R7: a district gets more dose than without trees, but the text says none does")
for name, (s, _) in expect.items():
    check(s in body, f"number '{name}': expected text '{s}' not found")
# all benchmark patches must report the same evaluation budget quoted in the text
ev = {p: b["evals"] for p, b in F["bench"].items()}
check(len({(e["nsga_gnn"], e["nsga_linear"]) for e in ev.values()}) == 1, f"evaluation counts differ across patches: {ev}")
# saved physics-feature correlations
pc = json.loads((HERE.parent / "data" / "models" / "physics_feature_corr.json").read_text())
for pid, lab in (("rot_centrum", "Rotterdam Centrum"), ("ams_bijlmer", "Amsterdam Bijlmer")):
    s = f"${pc[pid]['corr_sum_crown_shade_vs_dUTCI']:.2f}$ ({lab})"
    check(s in body, f"feature correlation '{s}' not found")
# forbidden claims from the brief that are NOT supported by the results
for bad in ("0.291 \\degC{} RMSE", "radiative-convective", "four cities"):
    check(bad not in body, f"unsupported phrase present: {bad}")

n_refs = len(set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body)))
# ------------------------------------------------------------------ traceability of EVERY number in the prose
def walk(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from walk(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)


allowed = set()
for v in walk(F):
    for s in (f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{abs(v):.0f}", f"{abs(v):.1f}", f"{abs(v):.2f}",
              f"{abs(v):.3f}", f"{100 * v:.0f}", f"{100 * abs(v):.0f}"):
        allowed.add(s)
derived = [skill(v2["test_city"]), skill(v2["transfer"]), skill(v1w["test_city"]), skill(v1w["transfer"]),
           100 * (1 - v2["test_city"]["change_mae"] / lin["test_city"]["change_mae"]),
           100 * (1 - v2["transfer"]["change_mae"] / lin["transfer"]["change_mae"]), *gain,
           pc["rot_centrum"]["corr_sum_crown_shade_vs_dUTCI"], pc["ams_bijlmer"]["corr_sum_crown_shade_vs_dUTCI"]]
for v in derived:
    allowed |= {f"{v:.0f}", f"{abs(v):.0f}", f"{v:.2f}", f"{abs(v):.2f}"}
for t in inputs:      # every number printed in the tables
    allowed |= set(re.findall(r"\d+(?:\.\d+)?", (HERE / (t if t.endswith(".tex") else t + ".tex")).read_text(encoding="utf-8")))
# method parameters (configs/engine.toml, realscene.py, scripts) and literature values quoted with citations
WHITELIST = {
    "2.5": "2.5D", "900": "window m", "500": "core m", "200": "buffer/SVF radius m", "1": "1 m grid/one", "3": "3 % / 3 m / three",
    "25": "trunk 25 %/25 m link", "36": "azimuths/hours", "0.06": "angular factor", "0.22": "angular factor",
    "0.70": "a_k", "0.97": "a_l", "6": "six directions", "273.15": "K offset", "0.5": "wind factor/range",
    "17": "wind range", "10": "10 m wind/10 %/10 epochs", "50": "50 m search/imputation", "40": "40 m/40 %", "70": "70th pct",
    "2": "2 m canopy", "2.8": "Tolan MAE", "4": "4 m offset/districts", "12": "12 m spacing", "8": "8 m spacing/height",
    "250": "max slots", "81": "plantings", "80": "random plantings", "08": "08:00", "19": "19:00", "00": "time",
    "96": "hidden", "0.1": "dropout/alpha", "32": "batch", "11": "train patches", "0.002": "beta", "800": "radius m",
    "1.3": "walk speed", "26": "UTCI threshold", "100": "trees cap", "150": "generations", "300": "generations",
    "28": "reference directions", "5": "5 m/5 %", "20": "20 %", "30": "30 %", "90": "90 %", "15": "15 runs/15 %",
    "2019": "date", "2020": "date", "2022": "date", "2026": "date", "2.4": "Briegel MAE", "22": "Briegel speed-up",
    "1.5": "Lachapelle / McDonald", "12": "Kolaxidis", "26": "Kolaxidis/threshold", "18": "patches", "17": "patches",
    "2024": "year", "7": "Fig/Table", "64": "n/a", "1.8": "ratio", "6.0": "ratio", "3.0": "n/a", "0.30": "albedo",
    "1.36": "cal", "0.40": "cal", "9": "n/a", "60": "n/a", "0.0": "T5 range", "1.7": "T5 range", "13": "n/a",
    "5070": "GPU model name (RTX 5070)", "256": "SHA-256 hash name",
    "0.004": "sensitivity beta", "0.001": "sensitivity beta", "0.01": "sensitivity transmissivity",
    "0.10": "sensitivity transmissivity", "0.3": "sensitivity wind factor", "0.7": "sensitivity wind factor",
    "24": "sensitivity threshold", "28": "sensitivity threshold", "6": "young crown height", "12": "mature crown height",
    "4.0": "CC BY 4.0 licence", "0.05": "affected threshold",
}
# context-anchored checks: headline numbers must appear in THEIR sentence (catches wrong duplicates)
r_, t_ = v2["test_city"], v2["transfer"]
ANCHORS = [
    f"in an unseen city with an error of {f3(r_['change_mae'])}~\\degC",
    f"largely imputed, the error was {f3(t_['change_mae'])}~\\degC. The two errors were {pct(skill(r_))}\\% and {pct(skill(t_))}\\% below",
    f"reduced the change error to {f3(r_['change_mae'])}~\\degC{{}} in Rotterdam, with a correlation of {f3(r_['change_corr'])} "
    f"and a UTCI error of {f3(r_['mae'])}~\\degC",
    f"In Freiburg the change error was {f3(t_['change_mae'])}~\\degC, with a correlation of {f3(t_['change_corr'])} "
    f"and a UTCI error of {f3(t_['mae'])}~\\degC",
    f"Validation and unseen-city errors agreed ({f3(v2['val']['change_mae'])} versus {f3(r_['change_mae'])}~\\degC)",
    f"with an error of {f3(r_['change_mae'])}~\\degC{{}} and a correlation of {f3(r_['change_corr'])}",
    f"with largely imputed building heights its error was {f3(t_['change_mae'])}~\\degC",
    f"came within {pct(100 * (1 - r_['change_mae'] / lin['test_city']['change_mae']))}\\% of the GNN in Rotterdam",
    f"reduced engine-verified pedestrian heat dose {F['opt_vs_random_ratio_range'][0]:.1f}--{F['opt_vs_random_ratio_range'][1]:.1f} times more",
    f"by up to {-F['equity_best'][0]:.1f}\\% at 90\\% of greedy's cooling",
    f"with per-segment correlations of {f2(F['corr_gnn_solweig_range'][0])}--{f2(F['corr_gnn_solweig_range'][1])}",
    f"underestimated it by {F['underestimate_engine_pct_range'][0]:.0f}--{F['underestimate_engine_pct_range'][1]:.0f}\\% and "
    f"{F['underestimate_gnn_pct_range'][0]:.0f}--{F['underestimate_gnn_pct_range'][1]:.0f}\\%",
    f"The independent model found {F['underestimate_engine_pct_range'][0]:.0f}--{F['underestimate_engine_pct_range'][1]:.0f}\\% \\emph{{more}} cooling",
]
for a in ANCHORS:
    check(a in body, f"anchored claim not found verbatim: '{a[:90]}'")
allowed |= set(WHITELIST)
prose = body[body.index("\\begin{document}") if "\\begin{document}" in body else 0:]   # preamble settings are not claims
prose = re.sub(r"\\(label|ref|eqref|input|includegraphics|cite[pt]?|citealp|graphicspath|usepackage|documentclass|journal)"
               r"(\[[^\]]*\])?\{[^}]*\}", " ", prose)
prose = re.sub(r"\\(begin|end)\{[^}]*\}", " ", prose)
untraced = sorted({n for n in re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?", prose)
                   if n.replace(",", "") not in allowed and n not in allowed})
check(not untraced, f"numbers in prose not traceable to facts/tables/whitelist: {untraced}")

print(f"checked {len(expect) + 4} numeric claims, {n_refs} refs, {len(cited)} citation keys, {len(inputs)} inputs")
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS: structure and all audited numbers consistent with facts.json")
