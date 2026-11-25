"""Publication figures and summary tables from the saved results (no re-running of models,
except the GNN predictions for Fig. 3, which are recomputed from the frozen checkpoint).

    .\\.venv\\Scripts\\python.exe scripts/10_figures_tables.py

Figures -> figures/*.pdf + *.png (300 dpi); tables -> tables/*.csv, *.md, *.tex.
Palette: validated categorical slots (light): blue #2a78d6, orange #eb6834, aqua #1baf7a;
reference/neutral in ink greys; every multi-series figure carries a legend AND direct labels
or distinct markers (aqua is < 3:1 on white -> relief by labels/markers).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heatgraph.config import DATA_DIR  # noqa: E402

FIG, TAB = ROOT / "figures", ROOT / "tables"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PATCH_LABEL = {"rot_centrum": "Rotterdam Centrum", "rot_kop_van_zuid": "Rotterdam Kop van Zuid",
               "rot_oude_westen": "Rotterdam Oude Westen", "xfer_freiburg_altstadt": "Freiburg Altstadt (transfer)"}
TEST = list(PATCH_LABEL)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


CITY = {"alm": "Almere", "amf": "Amersfoort", "ams": "Amsterdam", "dha": "Den Haag", "dlf": "Delft",
        "ein": "Eindhoven", "gro": "Groningen", "rot": "Rotterdam", "utr": "Utrecht", "xfer": ""}
DISTRICT = {"centrum": "Centrum", "vathorst": "Vathorst", "bijlmer": "Bijlmer", "bos_en_lommer": "Bos en Lommer",
            "grachtengordel": "Grachtengordel", "ijburg": "IJburg", "zuidas": "Zuidas",
            "beatrixkwartier": "Beatrixkwartier", "schilderswijk": "Schilderswijk", "campus": "Campus",
            "strijp_s": "Strijp-S", "binnenstad": "Binnenstad", "kop_van_zuid": "Kop van Zuid",
            "oude_westen": "Oude Westen", "leidsche_rijn": "Leidsche Rijn", "freiburg_altstadt": "Freiburg Altstadt"}
SHORT_MODEL = {"GATv2 attention, geometric edges": "GATv2, geometric",
               "Sum aggregation, geometric edges, within-patch val.": "Sum, geometric, planting val.",
               "Sum aggregation, geometric edges, leave-patch-out val.": "Sum, geometric, patch val.",
               "Linear model on crown-shadow features": "Linear, crown-shadow",
               "Sum aggregation, crown-shadow edges (ours)": "Sum, crown-shadow (ours)"}
MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def patch_name(pid):
    pre, rest = pid.split("_", 1)
    return f"{CITY[pre]} {DISTRICT[rest]}".strip()


def tex_cell(s):
    """LaTeX typesetting of one cell; digits are never changed (the numeric audit reads them)."""
    s = str(s)
    if s == "-":
        return "--"
    s = s.replace("%", r"\%").replace("_", r"\_").replace("°", r"$^\circ$").replace(" ± ", r" $\pm$ ")
    if len(s) > 1 and s[0] in "+-" and s[1].isdigit():
        sign, num = s[0], s[1:]
        s = num if float(num) == 0 else ("$-$" if sign == "-" else "$+$") + num
    return s


def write_table(name, header, rows, caption, tex=None):
    """CSV + Markdown keep the raw values; the LaTeX version is typeset for the manuscript.

    tex: optional dict with 'caption' (LaTeX), 'head' (header lines), 'colspec', 'size', 'tabcolsep',
    'first' (function for the first column), 'group' (print a repeated first column once per group) and
    'rows' (rows with typeset labels; numbers must equal those in `rows`)."""
    TAB.mkdir(exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    t = tex or {}
    lines = [rf"\begin{{table}}[{t.get('placement', 'tbp')}]", r"\centering", t.get("size", r"\small")]
    if "tabcolsep" in t:
        lines.append(rf"\setlength{{\tabcolsep}}{{{t['tabcolsep']}}}")
    lines += [rf"\caption{{{t.get('caption', tex_cell(caption))}}}", rf"\label{{tab:{name}}}",
              r"\begin{tabular}{" + t.get("colspec", "l" + "r" * (len(header) - 1)) + "}", r"\toprule"]
    lines += t.get("head", [" & ".join(tex_cell(h) for h in header) + r" \\"])
    lines.append(r"\midrule")
    first, prev = t.get("first", tex_cell), None
    for r in t.get("rows", rows):
        c0 = first(r[0])
        if t.get("group"):
            if prev is not None and c0 != prev:
                lines.append(r"\addlinespace")
            c0, prev = ("" if c0 == prev else c0), c0
        lines.append(" & ".join([c0] + [tex_cell(c) for c in r[1:]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table", name)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------- tables
def table_patches():
    rows = list(csv.DictReader(open(DATA_DIR / "raw" / "ingest_summary.csv", encoding="utf-8")))
    import tomllib
    cfg = tomllib.load(open(ROOT / "configs" / "patches.toml", "rb"))
    lcz = {p["id"]: p.get("lcz_intended", "") for k in ("patch", "transfer") for p in cfg.get(k, [])}
    role = lambda pid: "test (city)" if pid.startswith("rot_") else "transfer" if pid.startswith("xfer_") else \
        "validation" if pid in ("ams_bos_en_lommer", "ein_strijp_s", "amf_vathorst") else "train"
    out = [[r["patch"], lcz.get(r["patch"], ""), role(r["patch"]), r["source"], r["buildings"], r["core_bsf"],
            r["core_h_mean_aw"], r["core_n_tall"], r["core_water_frac"], r["canopy_gt2m_pct"], r["walk_km"]]
           for r in rows]
    pretty = {"3dbag": "3DBAG", "overture": "Overture", "test (city)": "test"}
    out_tex = [[r[0], r[1].replace(" (new)", ""), pretty.get(r[2], r[2]), pretty.get(r[3], r[3])] + r[4:] for r in out]
    write_table("T1_patches", ["Patch", "Intended LCZ", "Role", "Heights", "Buildings", "BSF", "h_aw (m)",
                               "Towers >40 m", "Water frac.", "Canopy >2 m (%)", "Walk net (km)"], out,
                "Study patches: 500 m core form metrics, data source and role in the experiments",
                tex=dict(size=r"\footnotesize", tabcolsep="3.5pt", first=patch_name, colspec="llll" + "r" * 7,
                         placement="!h",
                         caption=r"Study patches: morphology of the 500~m core, source of building heights and role in "
                                 r"the experiments. LCZ, Local Climate Zone; BSF, building surface fraction; "
                                 r"$h_\mathrm{aw}$, area-weighted building height. Test patches were used only for "
                                 r"evaluation; Freiburg is the transfer patch with imputed heights.",
                         head=[r" & & & & & & & Towers & Water & Canopy & Walking \\",
                               r"Patch & Intended LCZ & Role & Heights & Buildings & BSF & $h_\mathrm{aw}$ (m) & "
                               r"$>$40 m & fraction & $>$2 m (\%) & network (km) \\"], rows=out_tex))


def table_engine():
    c = jload(DATA_DIR / "validation" / "calibration.json")
    if not c:
        return
    rows = [["ams_grachtengordel 2019-07-25 (calibration)", f"{c['calibration']['before']['mae_all']:.2f}",
             f"{c['calibration']['before']['bias_all']:+.2f}", f"{c['calibration']['after']['mae_all']:.2f}",
             f"{c['calibration']['after']['bias_all']:+.2f}", "fit"]]
    for k, v in c["held_out"].items():
        rows.append([k.replace(":", " "), f"{v['default']['mae_all']:.2f}", f"{v['default']['bias_all']:+.2f}",
                     f"{v['calibrated']['mae_all']:.2f}", f"{v['calibrated']['bias_all']:+.2f}",
                     "pass" if v["calibrated"]["pass"] else "FAIL"])
    write_table("T2_engine_vs_solweig", ["Patch / day", "MAE default (K)", "Bias default (K)", "MAE calibrated (K)",
                                         "Bias calibrated (K)", "Gate"], rows,
                "Tmrt of the pure-Python engine vs SOLWEIG-GPU (all core ground pixel-hours); gate MAE <= 3 K and |bias| <= 1 K",
                tex=dict(first=patch_day, colspec="lrrrrl", tabcolsep="5pt",
                         caption=r"Mean radiant temperature of the 2.5D engine against SOLWEIG-GPU over all ground "
                                 r"pixel-hours of the 500~m core, with default and calibrated parameters. The first row "
                                 r"is the calibration patch-day (fit); the others are held out. Acceptance gate: "
                                 r"MAE $\le$ 3~K and $|\mathrm{bias}|$ $\le$ 1~K.",
                         head=[r" & \multicolumn{2}{c}{Default} & \multicolumn{2}{c}{Calibrated} & \\",
                               r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                               r"Patch and day & MAE (K) & Bias (K) & MAE (K) & Bias (K) & Gate \\"]))


def patch_day(label):
    """'ams_grachtengordel 2019-07-25 (calibration)' -> 'Amsterdam Grachtengordel, 25 Jul 2019 (calibration)'."""
    pid, day, *rest = label.split(" ", 2)
    y, m, d = day.split("-")
    return f"{patch_name(pid)}, {int(d)} {MONTH[int(m) - 1]} {y}"     # '(calibration)' is in the caption


def table_surrogates():
    m = DATA_DIR / "models"
    spec = [("GATv2 attention, geometric edges", "metrics_gatv2.json"),
            ("Sum aggregation, geometric edges, within-patch val.", "metrics_edgemlp.json"),
            ("Sum aggregation, geometric edges, leave-patch-out val.", "metrics_edgemlp_v1_patchval.json"),
            ("Linear model on crown-shadow features", "metrics_linear_physics.json"),
            ("Sum aggregation, crown-shadow edges (ours)", "metrics_edgemlp_v2_patchval.json")]
    rows = []
    for name, f in spec:
        r = jload(m / f)
        if not r:
            continue
        cells = [name]
        for k in ("test_city", "transfer"):
            x = r[k]
            cells += [f"{x['change_mae']:.3f}", f"{x['change_corr']:.3f}",
                      f"{100 * (1 - x['change_mae'] / x['zero_change_mae']):.0f}"]
        rows.append(cells)
    write_table("T3_surrogates", ["Model", "Rotterdam dMAE (°C)", "Rotterdam corr", "Rotterdam skill (%)",
                                  "Freiburg dMAE (°C)", "Freiburg corr", "Freiburg skill (%)"], rows,
                "Surrogate accuracy on the tree-induced UTCI change in unseen cities; skill = 1 - dMAE / zero-change MAE",
                tex=dict(colspec="lrrrrrr", tabcolsep="5pt", first=lambda s: SHORT_MODEL.get(s, s),
                         caption=r"Surrogate accuracy on the tree-induced change in segment UTCI in unseen cities. "
                                 r"Sum, sum-aggregation GNN; GATv2, attention GNN; geometric/crown-shadow, slot-to-segment edges "
                                 r"without/with the crown-shadow and under-crown features; planting/patch val., model selection on held-out plantings of the training patches or on whole held-out patches. MAE, "
                                 r"mean absolute error of the change against the engine; $r$, Pearson correlation; "
                                 r"skill $= 1 - \mathrm{MAE}/\mathrm{MAE}_0$, where $\mathrm{MAE}_0$ is the error of "
                                 r"predicting no change. Rotterdam is an unseen Dutch city with lidar-based heights "
                                 r"(3DBAG); Freiburg is an unseen German city whose heights come from another source "
                                 r"and are largely imputed, a shift in country and input regime.",
                         head=[r" & \multicolumn{3}{c}{Rotterdam} & \multicolumn{3}{c}{Freiburg} \\",
                               r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                               r"Model & MAE ($^\circ$C) & $r$ & Skill (\%) & MAE ($^\circ$C) & $r$ & Skill (\%) \\"]))
    return spec


def bench_results():
    out = {}
    for pid in TEST:
        r = jload(DATA_DIR / "stage5" / pid / "results.json")
        if r:
            rev = jload(DATA_DIR / "stage5" / pid / "revision_r2r3.json")   # practitioner heuristics, unseeded NSGA-III
            if rev:
                r = dict(r, rows=r["rows"] + rev["rows"], revision=rev)
            out[pid] = r
    return out


def table_benchmark(B):
    rows = []
    for pid, r in B.items():
        tab = {}
        for x in r["rows"]:
            tab.setdefault((x["method"], x["budget"]), []).append(x["engine_dose_reduction_pct"])
        heur = ("heur_sunflow", "heur_sun") if ("heur_sunflow", r["budgets"][0]) in tab else ()
        for b in r["budgets"]:
            rows.append([PATCH_LABEL[pid], f"{b} ({100 * b / r['n_slots']:.0f}%)"] +
                        [f"{np.mean(tab[(m, b)]):.2f}" for m in ("nsga_gnn", "nsga_linear", "greedy") + heur] +
                        [f"{np.mean(tab[('random', b)]):.2f} ± {np.std(tab[('random', b)]):.2f}"])
    if heur:
        header = ["Patch", "Trees (share of slots)", "NSGA-III GNN (%)", "NSGA-III linear (%)", "Greedy (%)",
                  "Heuristic sun x flow (%)", "Heuristic sun only (%)", "Random (%)"]
        cap = (r"Engine-verified reduction (\%) of person-weighted heat dose (UTCI $>$ 26~$^\circ$C) by method and "
               r"tree budget in the four test patches (Freiburg Altstadt is the transfer patch). The two practitioner "
               r"heuristics use no surrogate and no optimiser: they plant the streets with the largest unshaded share, "
               r"weighted by pedestrian flow (sun $\times$ flow) or not (sun only). Random layouts: mean $\pm$ "
               r"standard deviation of ten layouts.")
        head = [r" & & \multicolumn{2}{c}{NSGA-III} & & \multicolumn{2}{c}{Heuristic} & \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
                r"Patch & Trees (share) & GNN & Linear & Greedy & Sun $\times$ flow & Sun only & Random \\"]
        colspec, sep, size = "lrrrrrrr", "3pt", r"\footnotesize"
    else:
        header = ["Patch", "Trees (share of slots)", "NSGA-III GNN (%)", "NSGA-III linear (%)", "Greedy (%)",
                  "Random (%)"]
        cap = (r"Engine-verified reduction (\%) of person-weighted heat dose (UTCI $>$ 26~$^\circ$C) "
               r"by method and tree budget in the four test patches (Freiburg Altstadt is the "
               r"transfer patch). Random layouts: mean $\pm$ standard deviation of ten layouts.")
        head = [r" & & \multicolumn{2}{c}{NSGA-III} & & \\", r"\cmidrule(lr){3-4}",
                r"Patch & Trees (share of slots) & GNN & Linear & Greedy & Random \\"]
        colspec, sep, size = "lrrrrr", "5pt", r"\small"
    write_table("T4_benchmark", header, rows,
                "Engine-verified reduction of person-weighted heat dose (UTCI > 26 °C) by method and tree budget",
                tex=dict(group=True, colspec=colspec, tabcolsep=sep, size=size, first=lambda s: s.replace(" (transfer)", ""),
                         caption=cap, head=head))
    eq = []
    for pid, r in B.items():
        for e in r.get("equity_at_matched_cooling", []):
            eq.append([PATCH_LABEL[pid], e["budget"], f"{e['greedy_theil']:.4f}", f"{e['front_theil']:.4f}",
                       f"{e['theil_change_pct']:+.1f}", f"{e['cooling_kept_pct']:.0f}"])
    if eq:
        write_table("T5_equity", ["Patch", "Trees", "Theil greedy", "Theil best front", "Change (%)",
                                  "Cooling kept (%)"], eq,
                    "Equity available on the NSGA-III (GNN) front at >= 90 % of greedy's cooling (population proxy)",
                    tex=dict(group=True, colspec="lrrrrr", first=lambda s: s.replace(" (transfer)", ""),
                             caption=r"Between-district equity available on the NSGA-III (GNN) front at $\ge$ 90\% of "
                                     r"greedy's engine-verified cooling. Theil index of per-capita dose across four "
                                     r"districts, with building floor area as population proxy (lower is more equal).",
                             head=[r" & & \multicolumn{2}{c}{Theil index} & & \\",
                                   r"\cmidrule(lr){3-4}",
                                   r"Patch & Trees & Greedy & Best front & Change (\%) & Cooling kept (\%) \\"]))


SENS_LABEL = [("baseline", "Reference setting", "-"),
              ("transmissivity_0.01", "Crown transmissivity 0.01", "physics"),
              ("transmissivity_0.10", "Crown transmissivity 0.10", "physics"),
              ("crown_young_r2_h6", "Young crown (2 m radius, 6 m)", "physics"),
              ("crown_mature_r4_h12", "Mature crown (4 m radius, 12 m)", "physics"),
              ("wind_factor_0.3", "Street wind factor 0.3", "physics"),
              ("wind_factor_0.7", "Street wind factor 0.7", "physics"),
              ("beta_0.001", "Gravity decay 0.001 per m", "objective"),
              ("beta_0.004", "Gravity decay 0.004 per m", "objective"),
              ("threshold_24", "UTCI threshold 24 °C", "objective"),
              ("threshold_28", "UTCI threshold 28 °C", "objective")]


def table_sensitivity():
    S = jload(DATA_DIR / "stage5" / "revision_round2_sensitivity.json")
    if not S:
        return
    rng = lambda xs, f: f"{f.format(min(xs))}–{f.format(max(xs))}"
    rows = []
    for key, label, kind in SENS_LABEL:
        pats = S["scenarios"].get(key)
        if not pats:
            continue
        top, ratio, share, jac = [], [], [], []
        for v in pats.values():
            bs = sorted(v, key=int)
            top.append(v[bs[-1]]["greedy"])
            for b in bs:
                ratio.append(v[b]["greedy"] / v[b]["random_mean"])
                share.append(100 * v[b]["heur"] / v[b]["greedy"])
                if "jaccard_vs_baseline_greedy" in v[b]:
                    jac.append(v[b]["jaccard_vs_baseline_greedy"])
        rows.append([label, rng(top, "{:.2f}"), rng(ratio, "{:.1f}"), rng(share, "{:.0f}"),
                     rng(jac, "{:.2f}") if jac else "-"])
    write_table("T7_sensitivity", ["Setting", "Greedy reduction at 40 % of slots (%)", "Greedy / random",
                                   "Sun x flow heuristic, share of greedy (%)", "Slot overlap with reference greedy"],
                rows, "Sensitivity of the allocation results to fixed modelling choices (engine-verified, four test patches)",
                tex=dict(colspec="lrrrr", tabcolsep="4pt", size=r"\footnotesize",
                         first=lambda s: s.replace("°", r"$^\circ$"),
                         caption=r"Sensitivity of the allocation results to fixed modelling choices, as ranges over "
                                 r"the four test patches and five budgets (engine-verified). Physical settings "
                                 r"re-simulate the layouts optimised under the reference setting; objective settings "
                                 r"re-run greedy under the new objective, and the overlap is the Jaccard index of its "
                                 r"planted slots with the reference greedy layout. Greedy reduction is at 40\% of "
                                 r"slots. Gravity decay is $\beta$ of the pedestrian-flow model.",
                         head=[r"Setting & \makecell[r]{Greedy reduction\\at 40\% (\%)} & \makecell[r]{Greedy /\\random} "
                               r"& \makecell[r]{Heuristic share\\of greedy (\%)} & \makecell[r]{Slot overlap\\with reference} \\"]))


def table_solweig(V):
    if not V:
        return
    R4 = jload(DATA_DIR / "stage5" / "revision_r4_verification.json")
    rows = []
    for n, d in V["designs"].items():
        a = d.get("agreement", {})
        c = R4["designs"].get(n) if R4 else None
        if R4:
            du = ([f"{c[m]['mean_dUTCI_common_mask']:+.2f}" for m in ("solweig", "engine", "gnn")] +
                  [f"{c['engine']['spread_ratio_vs_solweig']:.2f}"]) if c else ["0.00", "0.00", "0.00", "-"]
        else:
            du = [f"{d[m]['mean_dUTCI_affected']:+.2f}" for m in ("solweig", "engine", "gnn")]
        rows.append([n.replace("_", " "), d["trees"], f"{d['solweig']['dose_reduction_pct']:.2f}",
                     f"{d['engine']['dose_reduction_pct']:.2f}", f"{d['gnn']['dose_reduction_pct']:.2f}"] + du +
                    [f"{a['corr_gnn_vs_solweig']:.2f}" if a else "-"])
    if R4:
        header = ["Design", "Trees", "Dose red. SOLWEIG (%)", "Dose red. engine (%)", "Dose red. GNN (%)",
                  "dUTCI SOLWEIG (°C)", "dUTCI engine (°C)", "dUTCI GNN (°C)", "Cooled share engine/SOLWEIG",
                  "corr(GNN, SOLWEIG)"]
        cap = (r"Cross-model verification of five Pareto designs with SOLWEIG-GPU in Rotterdam Centrum, an unseen "
               r"city. Dose reduction is relative to each model's own run without new trees. $\Delta$UTCI is the mean "
               r"over the segment-hours that SOLWEIG-GPU cools by more than 0.05~$^\circ$C, a mask common to all "
               r"three models. Cooled share: segment-hours the engine cools by more than 0.05~$^\circ$C, relative to "
               r"SOLWEIG-GPU. $r$ is the per-segment-hour correlation of $\Delta$UTCI between the GNN and SOLWEIG-GPU.")
        colspec, sep = "lrrrrrrrrr", "3pt"
        head = [r" & & \multicolumn{3}{c}{Dose reduction (\%)} & "
                r"\multicolumn{3}{c}{Mean $\Delta$UTCI ($^\circ$C)} & & \\",
                r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                r"Design & Trees & SOLWEIG & Engine & GNN & SOLWEIG & Engine & GNN & Cooled share & $r$ \\"]
    else:
        header = ["Design", "Trees", "Dose red. SOLWEIG (%)", "Dose red. engine (%)", "Dose red. GNN (%)",
                  "dUTCI SOLWEIG (°C)", "dUTCI engine (°C)", "dUTCI GNN (°C)", "corr(GNN, SOLWEIG)"]
        cap = (r"Independent SOLWEIG-GPU verification of five Pareto designs in Rotterdam Centrum, an "
               r"unseen city. Dose reduction is relative to each model's own run without new trees; "
               r"$\Delta$UTCI is the mean over affected segment-hours; $r$ is the per-segment-hour "
               r"correlation of $\Delta$UTCI between the GNN and SOLWEIG-GPU.")
        colspec, sep = "lrrrrrrrr", "4.5pt"
        head = [r" & & \multicolumn{3}{c}{Dose reduction (\%)} & "
                r"\multicolumn{3}{c}{Mean $\Delta$UTCI ($^\circ$C)} & \\",
                r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                r"Design & Trees & SOLWEIG & Engine & GNN & SOLWEIG & Engine & GNN & $r$ \\"]
    write_table("T6_solweig_verification", header, rows,
                "Cross-model SOLWEIG-GPU verification of five Pareto designs (Rotterdam Centrum, unseen city); "
                "dUTCI = mean over segment-hours SOLWEIG cools by > 0.05 °C (common mask)",
                tex=dict(colspec=colspec, tabcolsep=sep, first=lambda s: s[0].upper() + s[1:], caption=cap, head=head))


# --------------------------------------------------------------------------- figures
def fig_engine_validation():
    c = jload(DATA_DIR / "validation" / "calibration.json")
    if not c:
        return
    names = list(c["held_out"])
    lab = [n.split(":")[0].replace("_", " ") + "\n" + n.split(":")[1] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    y = np.arange(len(names))
    for k, (key, lim, xl) in enumerate([("mae_all", 3.0, "Tmrt MAE vs SOLWEIG-GPU (K)"),
                                        ("bias_all", 1.0, "Tmrt bias vs SOLWEIG-GPU (K)")]):
        d0 = [c["held_out"][n]["default"][key] for n in names]
        d1 = [c["held_out"][n]["calibrated"][key] for n in names]
        a = ax[k]
        for i in range(len(names)):
            a.plot([d0[i], d1[i]], [y[i], y[i]], color=AXIS, lw=1.2, zorder=1)
        a.scatter(d0, y, s=36, facecolor="white", edgecolor=INK2, lw=1.2, label="default", zorder=2)
        a.scatter(d1, y, s=36, color=BLUE, label="calibrated", zorder=3)
        if key == "mae_all":
            a.axvline(lim, color=ORANGE, lw=1, ls="--")
            a.text(lim, len(names) - 0.4, " gate 3 K", color=INK2, va="bottom", fontsize=7)
            a.set_xlim(0, 3.4)
        else:
            a.axvspan(-lim, lim, color=GRID, alpha=0.6, lw=0)
            a.axvline(0, color=AXIS, lw=0.8)
            a.text(-lim, len(names) - 0.4, " gate ±1 K", color=INK2, va="bottom", fontsize=7)
            a.set_xlim(-2.2, 0.6)
        a.set_yticks(y, lab if k == 0 else [])
        a.set_xlabel(xl)
        a.set_ylim(-0.6, len(names) - 0.2)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    ax[0].set_title("a  Accuracy on held-out patch-days", loc="left")
    ax[1].set_title("b  Bias", loc="left")
    save(fig, "F1_engine_validation")


def fig_surrogate_ablation(spec):
    m = DATA_DIR / "models"
    items = [(n, jload(m / f)) for n, f in spec]
    items = [(n, r) for n, r in items if r]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = np.arange(len(items))
    h = 0.36
    for k, (key, col, lab) in enumerate([("test_city", BLUE, "Rotterdam (unseen city)"),
                                         ("transfer", ORANGE, "Freiburg (unseen city, Overture heights)")]):
        v = [r[key]["change_mae"] for _, r in items]
        ax.barh(y + (k - 0.5) * h, v, height=h * 0.92, color=col, label=lab)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.004, yi + (k - 0.5) * h, f"{vi:.3f}", va="center", fontsize=7, color=INK2)
    z = np.mean([items[0][1]["test_city"]["zero_change_mae"], items[0][1]["transfer"]["zero_change_mae"]])
    ax.axvline(z, color=INK2, ls="--", lw=1)
    ax.text(z, len(items) - 0.45, " predict 'no change'", color=INK2, fontsize=7, va="bottom")
    ax.set_yticks(y, [n for n, _ in items])
    ax.set_xlabel("MAE of tree-induced ΔUTCI per segment-hour (°C)")
    ax.set_ylim(-0.6, len(items) - 0.1)
    ax.set_xlim(0, max(r[k]["change_mae"] for _, r in items for k in ("test_city", "transfer")) * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Surrogate ablation on unseen cities (lower is better)", loc="left")
    save(fig, "F2_surrogate_ablation")


def fig_parity():
    import torch

    from heatgraph.config import load_toml
    from heatgraph.realscene import load_scene
    from heatgraph.surrogate.graphs import SceneGraph
    from heatgraph.surrogate.io import load_gnn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_toml("engine.toml")
    ck = DATA_DIR / "models" / "heatgraph_edgemlp_v2_patchval.pt"
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for a, (title, pids) in zip(ax, [("a  Rotterdam (unseen city)", [p for p in TEST if p.startswith("rot_")]),
                                     ("b  Freiburg (other country, imputed heights)",["xfer_freiburg_altstadt"])]):
        dt, dp = [], []
        for pid in pids:
            s = load_scene(DATA_DIR / "processed" / pid)
            z = np.load(DATA_DIR / "processed" / pid / "labels.npz")
            X, U = z["plantings"], z["utci"]
            sg = SceneGraph(s, baseline_utci=U[0], cfg=cfg, device=dev)
            gnn = load_gnn(ck, sg.build(X[0], 0), dev)
            P = []
            with torch.no_grad():
                for i in range(1, len(X), 8):
                    P.append(gnn.predict(sg.build_batch(X[i:i + 8]).to(dev)).cpu().numpy().reshape(-1, sg.hours,
                                                                                                    s.n_segments))
            P = np.concatenate(P)
            dt.append((U[1:] - U[0]).ravel())
            dp.append((P - U[0]).ravel())
        dt, dp = np.concatenate(dt), np.concatenate(dp)
        lo = min(dt.min(), dp.min())
        hb = a.hexbin(dt, dp, gridsize=60, bins="log", mincnt=1, cmap=matplotlib.colors.ListedColormap(SEQ),
                      extent=(lo, 0.3, lo, 0.3), linewidths=0)
        a.plot([lo, 0.3], [lo, 0.3], color=ORANGE, lw=1, ls="--")
        a.text(0.04, 0.95, f"r = {np.corrcoef(dt, dp)[0, 1]:.3f}\nMAE = {np.mean(np.abs(dt - dp)):.3f} °C\n"
                           f"n = {len(dt):,}", transform=a.transAxes, va="top", fontsize=7.5, color=INK,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
        a.set_title(title, loc="left")
        a.set_xlabel("Engine ΔUTCI (°C)")
        a.set_aspect("equal")
    ax[0].set_ylabel("GNN ΔUTCI (°C)")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("segment-hours (log)", color=INK2)
    save(fig, "F3_parity_unseen_cities")


def fig_density():
    d = jload(DATA_DIR / "models" / "density_analysis.json")
    if not d:
        return
    x = [100 * (b["lo"] + b["hi"]) / 2 for b in d["by_bin"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for key, col, mk, lab in (("gnn", BLUE, "o", "GNN (crown-shadow edges)"), ("linear", ORANGE, "s", "Linear physics"),
                              ("zero", MUTED, "^", "No-change reference")):
        v = [b[key] for b in d["by_bin"]]
        ax.plot(x, v, color=col, marker=mk, ms=5, ls="--" if key == "zero" else "-", label=lab)
        ax.text(x[-1] + 1, v[-1], lab, color=INK2, fontsize=7, va="center")
    ax.set_xlim(0, 52)
    ax.set_xlabel("Share of slots planted (%)")
    ax.set_ylabel("ΔUTCI MAE (°C)")
    ax.set_title("Where the GNN beats the linear model", loc="left")
    ax.legend(loc="upper left")
    save(fig, "F4_error_by_density")


def fig_benchmark(B):
    if not B:
        return
    n = len(B)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n + 0.6, 2.7), sharey=False)
    axes = np.atleast_1d(axes)
    for a, (pid, r) in zip(axes, B.items()):
        tab = {}
        for x in r["rows"]:
            tab.setdefault((x["method"], x["budget"]), []).append(x["engine_dose_reduction_pct"])
        bx = [100 * b / r["n_slots"] for b in r["budgets"]]
        rnd = np.array([tab[("random", b)] for b in r["budgets"]])
        a.fill_between(bx, rnd.min(1), rnd.max(1), color=GRID, lw=0, label="random (range)")
        a.plot(bx, rnd.mean(1), color=MUTED, marker="^", ms=4, lw=1.2, label="random (mean)")
        if ("heur_sun", r["budgets"][0]) in tab:
            a.plot(bx, [np.mean(tab[("heur_sun", b)]) for b in r["budgets"]], color=INK2, marker="v", ms=4,
                   lw=1.0, ls=":", label="heuristic: sun only")
            a.plot(bx, [np.mean(tab[("heur_sunflow", b)]) for b in r["budgets"]], color=INK2, marker="D", ms=3.5,
                   lw=1.1, ls="--", label="heuristic: sun × flow")
        a.plot(bx, [np.mean(tab[("greedy", b)]) for b in r["budgets"]], color=AQUA, marker="o", ms=8,
               mfc="white", mew=1.6, lw=1.4, label="greedy (linear)")
        a.plot(bx, [np.mean(tab[("nsga_linear", b)]) for b in r["budgets"]], color=ORANGE, marker="s", ms=5,
               lw=1.2, label="NSGA-III linear")
        a.plot(bx, [np.mean(tab[("nsga_gnn", b)]) for b in r["budgets"]], color=BLUE, marker="o", ms=3.5,
               lw=1.2, label="NSGA-III GNN (ours)")
        a.set_title(PATCH_LABEL[pid].replace(" (transfer)", "\n(transfer)"), loc="left", fontsize=8)
        a.set_xlabel("Trees (% of slots)")
        a.set_xticks([5, 10, 20, 30, 40])
    axes[0].set_ylabel("Engine-verified heat-dose\nreduction (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4 if len(h) > 5 else 5, bbox_to_anchor=(0.5, -0.02))
    save(fig, "F5_benchmark_unseen_cities")


def fig_front_and_map(B, V):
    r = B.get("rot_centrum")
    if not r:
        return
    d0 = r["engine_dose_none"]
    F = r["front_gnn"]
    t = np.array([f["trees"] for f in F])
    red = 100 * (d0 - np.array([f["engine_dose"] for f in F])) / d0
    th = np.array([f["engine_theil"] for f in F])
    fig = plt.figure(figsize=(7.0, 3.1))
    a = fig.add_axes([0.07, 0.15, 0.40, 0.75])
    sc = a.scatter(t, red, c=th, s=9, cmap=matplotlib.colors.ListedColormap(SEQ[::-1]), lw=0)
    cb = fig.colorbar(sc, ax=a, pad=0.02)
    cb.set_label("Theil index of per-capita dose (proxy)", color=INK2)
    if V:
        z = np.load(DATA_DIR / "stage5" / "rot_centrum" / "solweig_verification_arrays.npz")
        for n, d in V["designs"].items():
            if n == "none":
                continue
            a.scatter(d["trees"], d["engine"]["dose_reduction_pct"], s=60, facecolor="none", edgecolor=ORANGE, lw=1.4,
                      zorder=3)
            off = (-58, 4) if d["trees"] > 80 else (-14, 9)
            a.annotate(n.replace("_", " "), (d["trees"], d["engine"]["dose_reduction_pct"]), xytext=off,
                       textcoords="offset points", fontsize=7, color=INK, ha="left",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    a.set_xlabel("Trees planted")
    a.set_ylabel("Engine-verified dose reduction (%)")
    a.set_title("a  NSGA-III (GNN) front, Rotterdam Centrum", loc="left")
    # map of the knee design
    b = fig.add_axes([0.58, 0.08, 0.40, 0.84])
    from heatgraph.realscene import load_scene

    s = load_scene(DATA_DIR / "processed" / "rot_centrum")
    H, W = s.building_heights.shape
    img = np.ones((H, W, 3))
    img[s.base_canopy > 0] = matplotlib.colors.to_rgb("#bfe8d6")
    img[s.water_mask] = matplotlib.colors.to_rgb("#e3eefb")
    img[s.building_heights > 0] = matplotlib.colors.to_rgb(AXIS)
    b.imshow(img, interpolation="nearest")
    if V:
        z = np.load(DATA_DIR / "stage5" / "rot_centrum" / "solweig_verification_arrays.npz")
        names = list(z["names"])
        k = names.index("knee")
        dU = (z["U_solweig"][k] - z["U_solweig"][0]).mean(0)          # mean over hours, per segment
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cool", ["#f0efec"] + SEQ[2:])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(0.5, float(-dU.min())))
        for e, rc in enumerate(s.receptors):
            b.plot(rc[:, 1], rc[:, 0], color=cmap(norm(-dU[e])), lw=1.6, solid_capstyle="round")
        x = z["X"][k].astype(bool)
        b.scatter(s.slots[x, 1], s.slots[x, 0], s=10, color=BLUE, edgecolor="white", lw=0.4, zorder=3)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb2 = fig.colorbar(sm, ax=b, shrink=0.7, pad=0.01)
        cb2.set_label("SOLWEIG cooling, mean −ΔUTCI (°C)", color=INK2)
    bb = 200
    b.set_xlim(bb, W - bb)
    b.set_ylim(H - bb, bb)
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_title("b  Knee design: new trees (blue) and cooling", loc="left")
    b.plot([bb + 20, bb + 120], [H - bb - 20] * 2, color=INK, lw=2)
    b.text(bb + 70, H - bb - 28, "100 m", ha="center", fontsize=7)
    save(fig, "F6_pareto_front_and_map")


def fig_solweig(V):
    if not V:
        return
    names = [n for n in V["designs"] if n != "none"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(names))
    w = 0.26
    for k, (m, col, lab) in enumerate([("solweig", INK2, "SOLWEIG-GPU (reference)"), ("engine", ORANGE, "Engine"),
                                       ("gnn", BLUE, "GNN surrogate")]):
        v = [V["designs"][n][m]["dose_reduction_pct"] for n in names]
        ax[0].bar(x + (k - 1) * w, v, width=w * 0.9, color=col, label=lab)
    ax[0].set_xticks(x, [f"{n.replace('_', ' ')}\n({V['designs'][n]['trees']} trees)" for n in names], fontsize=7)
    ax[0].set_ylabel("Heat-dose reduction (%)")
    ax[0].legend(loc="upper left")
    ax[0].set_title("a  Cooling of optimised layouts, three models", loc="left")
    z = np.load(DATA_DIR / "stage5" / "rot_centrum" / "solweig_verification_arrays.npz")
    nm = list(z["names"])
    k = nm.index("knee")
    ds = (z["U_solweig"][k] - z["U_solweig"][0]).ravel()
    dg = (z["U_gnn"][k] - z["U_gnn"][0]).ravel()
    de = (z["U_engine"][k] - z["U_engine"][0]).ravel()
    lo = min(ds.min(), dg.min(), de.min())
    ax[1].plot([lo, 0.2], [lo, 0.2], color=MUTED, ls="--", lw=1)
    ax[1].scatter(ds, de, s=6, color=ORANGE, marker="s", lw=0, alpha=0.6, label=f"engine (r={np.corrcoef(ds, de)[0, 1]:.2f})")
    ax[1].scatter(ds, dg, s=6, color=BLUE, lw=0, alpha=0.6, label=f"GNN (r={np.corrcoef(ds, dg)[0, 1]:.2f})")
    ax[1].set_xlabel("SOLWEIG-GPU ΔUTCI (°C)")
    ax[1].set_ylabel("Predicted ΔUTCI (°C)")
    ax[1].legend(loc="upper left", markerscale=2.5)
    ax[1].set_title("b  Knee design, per segment-hour", loc="left")
    save(fig, "F7_solweig_verification")


def main(tables_only=False):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    B = bench_results()
    V = jload(DATA_DIR / "stage5" / "rot_centrum" / "solweig_verification.json")
    table_patches()
    table_engine()
    spec = table_surrogates()
    table_benchmark(B)
    table_solweig(V)
    table_sensitivity()
    if tables_only:
        return
    fig_engine_validation()
    fig_surrogate_ablation(spec)
    fig_parity()
    fig_density()
    fig_benchmark(B)
    fig_front_and_map(B, V)
    fig_solweig(V)


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
