"""HeatGraph interactive dashboard (read-only view of the saved results).

    streamlit run app.py          ->  http://localhost:8501

Reads only dashboard_data/ (a small versioned extract of the results, built by
scripts/build_dashboard_data.py) and tables/*.csv, so it runs from a plain clone of the repository.
It never writes anything. All cooling values are ENGINE re-simulations (or SOLWEIG-GPU in the
verification tab), not surrogate predictions; tree-induced UTCI changes are stored at 0.001 degC.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dashboard_data"
PATCHES = {"rot_centrum": "Rotterdam Centrum", "rot_kop_van_zuid": "Rotterdam Kop van Zuid",
           "rot_oude_westen": "Rotterdam Oude Westen", "xfer_freiburg_altstadt": "Freiburg Altstadt (transfer)"}
# methods with a map (layouts + engine UTCI); the benchmark chart also shows unseeded NSGA-III
METHODS = {"nsga_gnn": "NSGA-III (GNN)", "nsga_linear": "NSGA-III (linear)", "greedy": "Greedy (linear)",
           "heur_sunflow": "Practitioner: sun × flow", "heur_sun": "Practitioner: sun only", "random": "Random"}
BENCH_METHODS = {**METHODS, "nsga_unseeded": "NSGA-III (GNN), no greedy seeding"}
COLORS = {"nsga_gnn": "#2a78d6", "nsga_linear": "#eb6834", "greedy": "#1baf7a", "heur_sunflow": "#52514e",
          "heur_sun": "#a8a7a0", "random": "#898781", "nsga_unseeded": "#9b59b6"}
BUFFER = 200
SEQ = ["#f0efec", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COOL = LinearSegmentedColormap.from_list("cool", SEQ)

st.set_page_config(page_title="HeatGraph explorer", layout="wide")


# --------------------------------------------------------------------------- data (cached, read-only)
@st.cache_data(show_spinner="Loading patch...")
def load_patch(pid: str) -> dict:
    s = np.load(DATA / pid / "scene.npz")
    w = pd.read_csv(DATA / pid / "weather.csv", parse_dates=["time_utc"])
    t = pd.DatetimeIndex(w["time_utc"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    tz = "Europe/Berlin" if pid.startswith("xfer_") else "Europe/Amsterdam"
    res = json.loads((DATA / pid / "results.json").read_text(encoding="utf-8"))
    d = np.load(DATA / pid / "designs.npz")
    rows = pd.DataFrame(res["rows"])
    front = pd.DataFrame(res["front_gnn"])
    d0 = res["engine_dose_none"]
    front["dose_reduction_pct"] = 100 * (d0 - front["engine_dose"]) / d0
    rev = json.loads((DATA / pid / "revision_r2r3.json").read_text(encoding="utf-8"))
    rev_rows = pd.DataFrame(rev["rows"]).assign(replicate=0)
    h = np.load(DATA / pid / "heuristic_layouts.npz")
    return {"rev_rows": rev_rows, "hv": rev["hypervolume_engine_normalised"],
            "h_names": list(h["names"]), "h_X": h["X"], "h_U": h["dU"].astype(np.float32) / float(h["scale"]),
            "building": s["building_heights"], "canopy": s["base_canopy"], "water": s["water_mask"],
            "receptors": s["receptors"], "seg": s["seg_of_receptor"], "slots": s["slots"],
            "local_time": t.tz_convert(tz), "rows": rows, "front": front, "X": d["X"], "front_X": d["front"],
            "U": d["dU"].astype(np.float32) / float(d["scale"]),   # UTCI change vs no new trees; U[0] == 0
            "budgets": res["budgets"], "n_slots": res["n_slots"], "d0": d0}


@st.cache_data
def load_solweig():
    p = DATA / "rot_centrum" / "solweig_verification_arrays.npz"
    if not p.exists():
        return None
    z = np.load(p)
    meta = json.loads((DATA / "rot_centrum" / "solweig_verification.json").read_text(encoding="utf-8"))
    return {"names": list(z["names"]), "X": z["X"], "solweig": z["U_solweig"], "engine": z["U_engine"],
            "gnn": z["U_gnn"], "meta": meta["designs"]}


@st.cache_data
def load_revision(name: str):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_table(name: str):
    p = ROOT / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def base_image(pid: str):
    P = load_patch(pid)
    img = np.ones(P["building"].shape + (3,))
    img[P["canopy"] > 0] = to_rgb("#bfe8d6")
    img[P["water"]] = to_rgb("#e3eefb")
    img[P["building"] > 0] = to_rgb("#c3c2b7")
    return img


# --------------------------------------------------------------------------- helpers
def hour_mask(P, choice: str) -> np.ndarray:
    lt = P["local_time"]
    if choice == "Mean of all 36 hours":
        return np.ones(len(lt), bool)
    day, hh = choice.split(" ")
    return np.asarray((lt.strftime("%Y-%m-%d") == day) & (lt.hour == int(hh[:2])))


def draw_map(P, pid, d_utci_seg, planted, title, vmax=None):
    """Core map: base layers, segments coloured by cooling (-dUTCI), planted slots as points."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(base_image(pid), interpolation="nearest")
    cool = -np.asarray(d_utci_seg)
    vmax = vmax or max(0.5, float(np.nanmax(cool)) if cool.size else 0.5)
    norm = Normalize(0, vmax)
    segs, cols = [], []
    for e in range(len(cool)):
        rc = P["receptors"][P["seg"] == e]
        if len(rc) >= 2:
            segs.append(rc[:, ::-1])
        elif len(rc) == 1:
            segs.append(np.vstack([rc[0, ::-1], rc[0, ::-1] + 0.5]))
        else:
            continue
        cols.append(COOL(norm(max(cool[e], 0))))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.2, capstyle="round"))
    if planted is not None and planted.any():
        xy = P["slots"][planted.astype(bool)]
        ax.scatter(xy[:, 1], xy[:, 0], s=14, color="#2a78d6", edgecolor="white", linewidth=0.5, zorder=3)
    H, W = P["building"].shape
    ax.set_xlim(BUFFER, W - BUFFER)
    ax.set_ylim(H - BUFFER, BUFFER)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([BUFFER + 15, BUFFER + 115], [H - BUFFER - 15] * 2, color="#0b0b0b", lw=2)
    ax.text(BUFFER + 65, H - BUFFER - 22, "100 m", ha="center", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=COOL)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("cooling, −ΔUTCI (°C)", fontsize=8)
    fig.tight_layout()
    return fig, vmax


def design_row(P, method, budget, replicate=0):
    r = P["rows"]
    m = r[(r["method"] == method) & (r["budget"] == budget) & (r["replicate"] == replicate)]
    return int(m.index[0])


def layout(P, method, budget, replicate=0):
    """(planted slots, UTCI change per hour and segment, benchmark row) for any mapped method."""
    if method.startswith("heur_"):
        j = P["h_names"].index(f"{method}_{budget}")
        r = P["rev_rows"]
        row = r[(r["method"] == method) & (r["budget"] == budget)].iloc[0]
        return P["h_X"][j], P["h_U"][j], row
    i = design_row(P, method, budget, replicate)
    return P["X"][i], P["U"][i], P["rows"].loc[i]


# --------------------------------------------------------------------------- UI
st.title("HeatGraph: exposure-weighted street-tree allocation")
st.caption("Read-only explorer of the saved study results. Cooling = engine re-simulation (calibrated 2.5D "
           "radiation + UTCI engine); the SOLWEIG tab shows the independent reference model. Equity uses a "
           "population proxy (building floor area), and pedestrian flows assume uniform trip generation.")

with st.sidebar:
    st.header("Selection")
    pid = st.selectbox("Patch (unseen test cities)", list(PATCHES), format_func=PATCHES.get)
    P = load_patch(pid)
    budget = st.select_slider("Tree budget", options=P["budgets"], value=P["budgets"][2],
                              format_func=lambda b: f"{b} trees ({100 * b / P['n_slots']:.0f}% of slots)")
    lt = P["local_time"]
    hour_opts = ["Mean of all 36 hours"] + [f"{t:%Y-%m-%d} {t:%H}:00" for t in lt]
    hour_choice = st.selectbox("Hour (local time)", hour_opts)
    st.divider()
    st.markdown("**Compare two layouts**")
    m_a = st.selectbox("Left map", list(METHODS), index=0, format_func=METHODS.get)
    m_b = st.selectbox("Right map", list(METHODS), index=3, format_func=METHODS.get)
    rep = st.slider("Random layout replicate", 0, 9, 0)

hm = hour_mask(P, hour_choice)
none_U = P["U"][0]
tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs(["Placements", "Benchmark", "Pareto front & equity", "SOLWEIG check",
                                              "Robustness", "Data & models"])

with tab1:
    cols = st.columns(2)
    vmax = None
    figs = []
    for col, m in zip(cols, (m_a, m_b)):
        x, U, row = layout(P, m, budget, rep if m == "random" else 0)
        d_seg = U[hm].mean(0)                     # U is already the change relative to no new trees
        figs.append((col, m, x, row, d_seg))
        vmax = max(vmax or 0, float(np.nanmax(-d_seg)))
    for col, m, x, row, d_seg in figs:
        with col:
            fig, _ = draw_map(P, pid, d_seg, x, f"{METHODS[m]}, {budget} trees", vmax=max(vmax, 0.5))
            st.pyplot(fig, clear_figure=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Heat-dose reduction", f"{row.engine_dose_reduction_pct:.2f}%")
            c2.metric("Theil (districts)", f"{row.engine_theil:.4f}")
            c3.metric("Trees", int(row.trees))
    st.caption("Segments are coloured by the engine's tree-induced cooling relative to no new trees, for the "
               "selected hour. Blue points are new trees; green areas existing canopy; grey buildings.")
    same = (P["X"][design_row(P, "nsga_gnn", budget)] == P["X"][design_row(P, "greedy", budget)]).all()
    st.info(f"At {budget} trees, the NSGA-III (GNN) and greedy layouts are "
            f"{'identical' if same else 'different'} on this patch.")

with tab2:
    r = pd.concat([P["rows"], P["rev_rows"]], ignore_index=True)
    r = r[r["method"] != "none"].copy()
    r["share"] = 100 * r["budget"] / P["n_slots"]
    agg = r.groupby(["method", "share"])["engine_dose_reduction_pct"].agg(["mean", "min", "max"]).reset_index()
    agg["Method"] = agg["method"].map(BENCH_METHODS)
    color = alt.Color("Method:N", scale=alt.Scale(domain=list(BENCH_METHODS.values()),
                                                  range=[COLORS[k] for k in BENCH_METHODS]))
    band = alt.Chart(agg[agg["method"] == "random"]).mark_area(opacity=0.25, color="#c3c2b7").encode(
        x="share:Q", y="min:Q", y2="max:Q")
    lines = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X("share:Q", title="Trees (% of candidate slots)"),
        y=alt.Y("mean:Q", title="Engine-verified heat-dose reduction (%)"), color=color,
        tooltip=["Method", alt.Tooltip("share:Q", format=".0f"), alt.Tooltip("mean:Q", format=".2f")])
    st.altair_chart((band + lines).properties(height=380), width="stretch")
    piv = r.groupby(["budget", "method"])["engine_dose_reduction_pct"].mean().unstack()[list(BENCH_METHODS)]
    piv.columns = [BENCH_METHODS[c] for c in piv.columns]
    st.dataframe(piv.style.format("{:.2f}"), width="stretch")
    g = piv[BENCH_METHODS["greedy"]]
    shf = 100 * piv[BENCH_METHODS["heur_sunflow"]] / g
    shs = 100 * piv[BENCH_METHODS["heur_sun"]] / g
    c1, c2, c3 = st.columns(3)
    c1.metric("Practitioner sun × flow, share of greedy", f"{shf.min():.0f}–{shf.max():.0f}%")
    c2.metric("Practitioner sun only, share of greedy", f"{shs.min():.0f}–{shs.max():.0f}%")
    c3.metric("Front hypervolume, unseeded vs seeded NSGA-III",
              f"{100 * (P['hv']['unseeded'] / P['hv']['seeded'] - 1):+.1f}%")
    st.caption("Random = mean of 10 layouts (band: range). NSGA-III and greedy largely coincide. The practitioner "
               "rules plant the sunniest streets without existing canopy, weighted by pedestrian flow or not; "
               "NSGA-III without greedy seeding starts from random layouts. See the paper, Section 4.4.")

with tab3:
    F = P["front"].copy()
    F = F[F["trees"] <= max(P["budgets"])].reset_index(drop=True)
    F["id"] = F.index
    sc = alt.Chart(F).mark_circle(size=28).encode(
        x=alt.X("trees:Q", title="Trees planted"), y=alt.Y("dose_reduction_pct:Q", title="Dose reduction (%)"),
        color=alt.Color("engine_theil:Q", title="Theil", scale=alt.Scale(scheme="blues", reverse=True)),
        tooltip=["id", "trees", alt.Tooltip("dose_reduction_pct:Q", format=".2f"),
                 alt.Tooltip("engine_theil:Q", format=".4f")])
    left, right = st.columns([1.2, 1])
    with left:
        st.altair_chart(sc.properties(height=420).interactive(), width="stretch")
        st.caption("NSGA-III (GNN) front, engine-verified. Darker = more equal between districts (population proxy).")
    with right:
        sub = F[F["trees"] <= budget]
        mode = st.radio("Pick a front layout within the budget", ["max cooling", "most equal at ≥90% of greedy cooling",
                                                                 "by id"], horizontal=False)
        g = P["rows"].loc[design_row(P, "greedy", budget)]
        if mode == "max cooling":
            j = int(sub["dose_reduction_pct"].idxmax())
        elif mode.startswith("most equal"):
            ok = sub[sub["dose_reduction_pct"] >= 0.9 * g.engine_dose_reduction_pct]
            j = int(ok["engine_theil"].idxmin()) if len(ok) else int(sub["dose_reduction_pct"].idxmax())
        else:
            j = st.number_input("Front layout id", 0, len(F) - 1, int(sub["dose_reduction_pct"].idxmax()))
        k = len(P["X"]) + int(j)
        fig, _ = draw_map(P, pid, (P["U"][k][hm] - none_U[hm]).mean(0), P["front_X"][j],
                          f"Front layout {j}: {int(F.loc[j, 'trees'])} trees")
        st.pyplot(fig, clear_figure=True)
        c1, c2 = st.columns(2)
        c1.metric("Dose reduction", f"{F.loc[j, 'dose_reduction_pct']:.2f}%",
                  f"{F.loc[j, 'dose_reduction_pct'] - g.engine_dose_reduction_pct:+.2f} pp vs greedy")
        c2.metric("Theil", f"{F.loc[j, 'engine_theil']:.4f}", f"{F.loc[j, 'engine_theil'] - g.engine_theil:+.4f} vs greedy",
                  delta_color="inverse")
    t5 = load_table("T5_equity")
    if t5 is not None:
        st.dataframe(t5[t5["Patch"] == PATCHES[pid]], width="stretch", hide_index=True)
    EQ = load_revision("revision_round2_equity")
    if EQ:
        st.markdown("**Per-district dose reduction: greedy vs the equity-selected layout**")
        rows_ = []
        for e in EQ["patches"][pid]["rows"]:
            order = np.argsort(e["per_capita_dose_none"])[::-1]
            for rank, k in enumerate(order):
                rows_.append({"Trees": e["budget"], "District (quadrant)": int(k),
                              "Dose rank (1 = highest)": rank + 1,
                              "Per-capita dose, no new trees": e["per_capita_dose_none"][k],
                              "Reduction greedy (%)": e["reduction_pct_greedy"][k],
                              "Reduction equity layout (%)": e["reduction_pct_equity"][k]})
        D = pd.DataFrame(rows_)
        D["Difference (pp)"] = D["Reduction equity layout (%)"] - D["Reduction greedy (%)"]
        st.dataframe(D[D["Trees"] == budget].style.format(
            {"Per-capita dose, no new trees": "{:.4f}", "Reduction greedy (%)": "{:.2f}",
             "Reduction equity layout (%)": "{:.2f}", "Difference (pp)": "{:+.2f}"}), width="stretch", hide_index=True)
        st.caption("Districts are the four quadrants of the 500 m core; population is a floor-area proxy. Across "
                   "patches and budgets, the more equal layouts mostly give the least exposed district less cooling "
                   "rather than giving the most exposed district more, and their size varies between optimiser "
                   "seeds (Robustness tab).")

with tab4:
    S = load_solweig()
    if pid != "rot_centrum" or S is None:
        st.info("The independent SOLWEIG-GPU verification was run for Rotterdam Centrum only "
                "(a limitation discussed in the paper). Select that patch to view it.")
    else:
        names = [n for n in S["names"] if n != "none"]
        n = st.selectbox("Verified layout", names, index=1, format_func=lambda x: x.replace("_", " "))
        i = S["names"].index(n)
        cols = st.columns(3)
        deltas = {m: (S[m][i][hm] - S[m][0][hm]).mean(0) for m in ("solweig", "engine", "gnn")}
        vmax = max(float(np.nanmax(-v)) for v in deltas.values())
        for col, (m, lab) in zip(cols, (("solweig", "SOLWEIG-GPU (reference)"), ("engine", "Engine"),
                                        ("gnn", "GNN surrogate"))):
            with col:
                fig, _ = draw_map(P, pid, deltas[m], S["X"][i], lab, vmax=vmax)
                st.pyplot(fig, clear_figure=True)
                st.metric("Dose reduction", f"{S['meta'][n][m]['dose_reduction_pct']:.2f}%")
        R4 = load_revision("revision_r4_verification")
        if R4 and n in R4["designs"]:
            c = R4["designs"][n]
            st.markdown("**Mean UTCI change on the segment-hours SOLWEIG cools by > 0.05 °C (common mask)**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("SOLWEIG-GPU", f"{c['solweig']['mean_dUTCI_common_mask']:+.2f} °C")
            k2.metric("Engine", f"{c['engine']['mean_dUTCI_common_mask']:+.2f} °C",
                      f"{c['engine']['bias_common_mask']:+.2f} vs SOLWEIG", delta_color="off")
            k3.metric("GNN", f"{c['gnn']['mean_dUTCI_common_mask']:+.2f} °C",
                      f"{c['gnn']['bias_common_mask']:+.2f} vs SOLWEIG", delta_color="off")
            k4.metric("Engine cooled share of SOLWEIG's segment-hours", f"{c['engine']['spread_ratio_vs_solweig']:.2f}")
            st.caption("Under the new crowns (SOLWEIG cooling > 1 °C) the engine cools more than SOLWEIG, by "
                       f"{-c['engine']['bias_strong_cooling']:.2f} °C on average; elsewhere it cools less. Totals are "
                       "conservative; single streets under new crowns are not.")
        a = S["meta"][n]["agreement"]
        st.write(f"Per segment-hour correlation with SOLWEIG: engine {a['corr_engine_vs_solweig']:.2f}, "
                 f"GNN {a['corr_gnn_vs_solweig']:.2f}. Each model's change is relative to its own no-planting run.")

with tab6:
    t7 = load_table("T7_sensitivity")
    if t7 is not None:
        st.markdown("**Sensitivity to modelling choices (Table 7), ranges over four patches and five budgets**")
        st.dataframe(t7, width="stretch", hide_index=True)
        st.caption("Physical settings re-simulate the reference layouts with the changed engine; objective settings "
                   "re-run greedy under the changed objective. Crown size changes the size of the benefit most; "
                   "every comparison between methods holds.")
    SN = load_revision("revision_round2_sensitivity")
    if SN:
        rows_ = []
        for name, pats in SN["scenarios"].items():
            if pid in pats:
                for b, v in pats[pid].items():
                    rows_.append({"Setting": name, "Trees": int(b), "Greedy (%)": v["greedy"],
                                  "Practitioner sun × flow (%)": v["heur"], "Random mean (%)": v["random_mean"]})
        S6 = pd.DataFrame(rows_)
        st.markdown(f"**This patch at {budget} trees: engine-verified dose reduction per setting**")
        st.dataframe(S6[S6["Trees"] == budget].drop(columns="Trees").style.format(
            {c: "{:.2f}" for c in S6.columns if "%" in c}), width="stretch", hide_index=True)
    CV = load_revision("revision_round2_coverage")
    if CV:
        st.markdown("**Coverage of the 90% conformal margin (GNN), against the engine**")
        cv = pd.DataFrame([{"Patch": PATCHES[k], "All test plantings": v["plantings"],
                            "Optimised layouts": v["optimised"],
                            "Worst single hour (optimised)": v["optimised_by_hour_min"]}
                           for k, v in CV["patches"].items()])
        st.dataframe(cv.style.format({c: "{:.1%}" for c in cv.columns if c != "Patch"}), width="stretch",
                     hide_index=True)
        st.caption("The margin is one value per hour, so it calibrates the heat-stress threshold rather than "
                   "penalising uncertain layouts. It reaches its nominal 90% on test plantings but not on the layouts "
                   "the optimiser picked, which is why every reported outcome is re-simulated.")
    SD = load_revision("revision_round2_seeds")
    if SD:
        v = SD["patches"][pid]
        st.markdown("**NSGA-III (GNN) over five random seeds, this patch**")
        rows_ = []
        for b in v["budgets"]:
            x = [r_["dose_reduction_pct"][str(b)] for r_ in v["runs"]]
            t = [r_["theil_change_pct"][str(b)] for r_ in v["runs"]]
            t = [z for z in t if z is not None]
            rows_.append({"Trees": b, "Greedy (%)": v["greedy_pct"][str(b)], "NSGA-III mean (%)": float(np.mean(x)),
                          "NSGA-III SD (pp)": float(np.std(x, ddof=1)),
                          "Theil change of best equity layout, per seed (%)": ", ".join(f"{z:+.1f}" for z in t) or "none"})
        st.dataframe(pd.DataFrame(rows_).style.format({"Greedy (%)": "{:.2f}", "NSGA-III mean (%)": "{:.2f}",
                                                        "NSGA-III SD (pp)": "{:.3f}"}),
                     width="stretch", hide_index=True)
        st.caption("Cooling is stable across seeds; the equity gain is not. Equity here searches a 100-layout front "
                   "subsample per seed, so values differ from Table 5, which uses up to 300 layouts.")

with tab5:
    for name, cap in (("T1_patches", "Study patches (core form metrics)"),
                      ("T2_engine_vs_solweig", "Engine vs SOLWEIG-GPU calibration gate"),
                      ("T3_surrogates", "Surrogate accuracy in unseen cities"),
                      ("T4_benchmark", "Engine-verified dose reduction by method, including practitioner rules"),
                      ("T5_equity", "Equity available on the front at >= 90% of greedy's cooling"),
                      ("T6_solweig_verification", "Cross-model SOLWEIG-GPU verification (common mask)"),
                      ("T7_sensitivity", "Sensitivity to modelling choices")):
        t = load_table(name)
        if t is not None:
            st.markdown(f"**{cap}**")
            st.dataframe(t, width="stretch", hide_index=True)
