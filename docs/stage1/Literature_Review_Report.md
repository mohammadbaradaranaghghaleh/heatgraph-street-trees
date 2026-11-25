# HeatGraph — Stage 1 Literature Review (narrative/integrative, `lit-review` mode)

*academic-pipeline Stage 1 · deep-research `lit-review` · strategist Phase 2 · 2026-11-24*
*Review form chosen by the author: narrative/integrative review (not systematic).*

## 1. Scope and question

**Working RQ.** Can a graph surrogate of pedestrian thermal comfort, trained on a pure-Python 2.5D radiation/UTCI engine, support tree allocations that minimise person-weighted heat dose on pedestrian networks under an explicit equity objective, and how well do those decisions transfer across cities and data sources?

**In scope:** pedestrian-level radiant heat and UTCI modelling; tree cooling and placement optimisation; neural and graph surrogates of urban microclimate; pedestrian flows, exposure and routing; heat and canopy equity; open data needed to run the pipeline.
**Out of scope:** air-temperature-only UHI mapping, building energy, CFD wind comfort (except as surrogate precedents), indoor comfort.

## 2. Search and verification method (reproducible)

| Step | What was done |
|---|---|
| Discovery | Targeted web searches (Nov 2026) across the six themes, plus snowballing from the closest works (Freiburg U-Net/placement, UrbanGraph, Lachapelle et al.) and foundational methods known in the field. |
| Inclusion | Peer-reviewed article, conference paper, or arXiv preprint for 2023+ ML work; directly informs a pipeline block or a gap; version of record preferred over preprint. |
| Verification | `verify_refs.py`: each candidate matched against **Crossref** (title + first-author surname, title similarity ≥ 0.85) or the **arXiv API** (id, title, author). Preprint DOIs returned by free-text search were replaced by **pinned version-of-record DOIs**, each re-fetched and checked. Gray zone = FAIL. |
| Result | 56 candidates screened over three passes → **50 PASS** (`references_verified.json`, `references_verified.md`). Excluded: one stub that matched an unrelated 2026 paper (the 2024 *Building and Environment* graph-attention comfort study could not be located and is **not** cited); five lower-priority classics trimmed to stay within the 35–50 target. |
| Gap check | `research_gaps.json` → `gap_analysis.py` (Quality Gate 2) → **PASS** (`research_gaps_gap_report.md`). A script confirms that every DOI cited as gap evidence is in the verified corpus. |

**Corpus by theme:** T1 thermal comfort and radiation 13 · T2 trees and placement 9 · T3 surrogates/XAI/UQ/optimisation 11 · T4 pedestrian networks and exposure 5 · T5 equity 7 · T6 open data 5. Eight carry arXiv DOIs: UrbanGraph (also an ICLR 2026 paper), six ML method papers from 2019–2021, and two recent applied preprints (Schrodi et al., 2023; Vu, 2025).

## 3. Thematic synthesis

### T1 — Radiant heat at pedestrian level is a solved physics problem, but an expensive one
Mean radiant temperature, not air temperature, dominates daytime outdoor heat stress in sunny streets, and the standard way to compute it at street scale is the six-directional flux scheme of [thorsson2007] as implemented in SOLWEIG [lindberg2008], later extended with vegetation shadowing [lindberg2011] and packaged in UMEP [lindberg2018]. Crown transmissivity values for trees come from field work such as [konarska2014]. Raster shadow casting on height grids [ratti2004] is the computational core that makes city-scale runs feasible. UTCI has a standardised operational procedure [brode2012] with an open, vectorised implementation [tartarini2020]; solar geometry is standard [reda2004; holmgren2018]. Field studies show that shade type and quality matter strongly for comfort [middel2016; middel2021]. A scriptable, GPU-accelerated SOLWEIG has now been published [solweig_gpu], which matters for this project: it provides an independent reference for validating a lighter pure-Python engine without depending on QGIS.
**Implication for HeatGraph:** the engine does not need new physics. It needs a faithful, fast re-implementation of established equations, validated against SOLWEIG-GPU before any learning.

### T2 — Trees cool people mostly through shade, and where they stand matters
Reviews and meta-analyses establish that urban greening, and trees in particular, cool streets and reduce heat stress, with effect size depending on species traits and context [bowler2010; rahman2020]. Tree canopy reduces surface temperature much more than other urban green [schwaab2021], and canopy and impervious cover interact non-linearly [ziter2019]. At pedestrian scale, trees reduce human heat stress more than grass [lee2016], and their location and arrangement change microclimate and comfort [zhao2018]. Most decisively for placement, a tree on an unshaded street yields roughly 1.5–2 times the pedestrian radiative cooling of one placed where shade already exists [philly_lup]. Optimisation work has started to exploit this: a neural Tmrt approximation made tree-placement search tractable over large areas [arxiv:2310.05691], and a deep-learning surrogate with Pareto search has been used for roof greening and cooling allocation [kim2025].
**Tension:** placement studies optimise heat over space, while the strongest pedestrian evidence [philly_lup] argues for place-based, person-relevant choices. This is the opening for GAP-002.

### T3 — Surrogates are accurate on average; their reliability for decisions is unmeasured
A U-Net emulating SOLWEIG Tmrt reached about 2.4 K MAE on held-out areas and days in Freiburg at about 22 times the speed [briegel2023], and was then embedded in a city-scale thermal-comfort emulator [briegel2024]. Graph models have arrived: GNNs estimate street-level air temperature consistently across locations [yu2024], and UrbanGraph encodes shading and convection in a dynamic heterogeneous graph for microclimate prediction [arxiv:2510.00457]. The methodological toolkit for a decision-grade graph surrogate exists: mesh/graph simulators [arxiv:2010.03409], attention backbones [arxiv:2105.14491], a standard library [arxiv:1903.02428], post-hoc explanation [arxiv:1903.03894], distribution-free uncertainty [arxiv:2107.07511], and many-objective search [deb2014; blank2020].
**Divergence:** all surrogate papers in the corpus report accuracy on observed or held-out scenes; none reviewed reports accuracy on the edited scenes an optimiser creates, uses calibrated uncertainty inside the objective, or re-simulates the optimised designs (GAP-001). A GNN alone is therefore not the contribution. Decision reliability is.

### T4 — Exposure happens along routes, and flows can be estimated
Personal heat exposure departs from fixed-station or area estimates [kuras2017]. Pedestrian flows on every segment can be estimated from network structure, destinations and detour behaviour, calibrated with counts [sevtsuk2021], and street networks are openly obtainable [boeing2017]. Shade-aware routing cuts solar exposure by about 12–26% in most tested cases and identifies streets where added shade would matter most [heidelberg_route]; street-view-based heat estimates have been attached to pedestrian network nodes for heat-aware routing [arxiv:2512.11896]. These studies take urban form as fixed. The step from "which route is cooler" to "where should trees go so that walked routes become cooler" has not been taken (GAP-002).

### T5 — Inequity is documented, not optimised
Canopy follows income across US cities [schwarz2015]; low-income blocks have about 15% less tree cover and are about 1.5 °C hotter [mcdonald2021]; heat-island exposure is higher for lower-income residents and people of colour in most large cities [hsu2021; chakraborty2019; benz2021]; and present canopy tracks historical redlining [locke2021]. Post-hoc equity analysis of mitigation infrastructure exists [sanchez2019]. Evidence is overwhelmingly US-based and descriptive or associational, and equity rarely enters the allocation algorithm itself (GAP-003).
**Caution for the paper:** transferring US equity findings to Dutch cities is not justified. HeatGraph should measure distributional outcomes locally rather than assume the US pattern.

### T6 — Open data make the pipeline possible, with known error budgets
LiDAR-derived LoD2/LoD1 models exist for all Dutch buildings [peters2022]; global 1 m canopy heights have about 2.8 m MAE [tolan2024]; a global Local Climate Zone map [demuzere2022; stewart2012] supports morphology-stratified sampling; ERA5 provides forcing [hersbach2020]. The canopy error is large relative to street-tree heights, and canopy features dominate street-level temperature models [yu2024], so input-data quality is a first-order risk for decisions (GAP-004).

## 4. Research gaps (Quality Gate 2: PASS)

| ID | Gap | Type | Significance | Evidence |
|---|---|---|---|---|
| GAP-001 | Surrogates validated on held-out scenes, not on optimiser edits; no calibrated uncertainty in the loop | partial | high | 6 |
| GAP-002 | Placement objectives ignore where people walk (no person-weighted dose on networks) | complete* | high | 6 |
| GAP-003 | Heat and canopy inequity documented but not optimised | partial | high | 6 |
| GAP-004 | Transferability and input-data quality untested for decision surrogates | partial | medium | 6 |

\*"Complete" is a claim about the reviewed corpus. A narrative review cannot prove absence; the claim must be re-tested with a Scopus/WoS query before submission (§6).

Full definitions, evidence and rationale: `research_gaps.json`; validator output: `research_gaps_gap_report.md`.

## 5. Originality assessment (strategist Step 2.3)

| Closest work | Overlap | HeatGraph difference |
|---|---|---|
| Schrodi et al. 2023 (surrogate + placement search) | High on mechanism | Graph representation; person-weighted network objective; equity objective; counterfactual-edit validation; conformal objective |
| Briegel et al. 2023/2024 (U-Net Tmrt emulator) | High on surrogate idea | Decision use; edits; transfer across cities and data sources |
| UrbanGraph 2025/26; Yu et al. 2024 (GNN microclimate) | Medium (graph prediction) | Decision-grade reliability, not predictive accuracy |
| Kolaxidis et al. 2025; Vu 2025 (heat-aware routing) | Medium (network exposure) | Interventions are optimised, not routes |
| Kim et al. 2025 (surrogate + Pareto roofs) | Medium (method) | Pedestrian scale, trees, equity, uncertainty |

**Estimated similarity:** about 45–50% with the closest single work, which falls in the "moderate, emphasise differences" band. **Innovation types:** methodological (decision-grade surrogate protocol) and integrative (thermal physics, pedestrian flow modelling and equity optimisation). **Impact prediction:** gap importance 4/5 · generalisability 2.5/3 · explanatory power 1.5/2 → **8/10** (unchanged from the idea note, but now evidence-backed).

## 6. Devil's-advocate check (Checkpoint 2) and limitations

- **Confirmation bias.** Searches targeted the gaps being claimed. Mitigation before Stage 2: run a counter-search for "tree placement" + ("pedestrian" OR "exposure" OR "population") and "surrogate" + ("counterfactual" OR "intervention" OR "out-of-distribution") in Scopus/WoS, and record hits that weaken GAP-001 or GAP-002.
- **Read scope.** Findings were taken from abstracts, publisher pages and search summaries, not full texts. Items tagged `[to verify]` in the JSON need full-text confirmation, in particular whether Schrodi et al. (2023) re-simulated optimised layouts and whether Kim et al. (2025) quantified surrogate uncertainty.
- **Quotes.** Evidence entries are **paraphrases marked `[paraphrase]`**, not verbatim quotes, so that copyrighted text is not reproduced at length. Gate 2 checks that the evidence fields are present, not that they are verbatim. Before submission, replace each with an exact sentence and page number from the paper.
- **Metadata quirks.** Crossref metadata is used as returned. Some issue years are online-first (e.g. Bröde et al. listed as 2011) and some author names are inverted (Holmgren et al.). Fix these in the reference manager, not by editing the JSON.
- **Geography.** The equity literature is US-dominated. Claims about Dutch cities must come from HeatGraph's own results.
- **Not systematic.** This is a narrative/integrative review by the author's choice. It does not claim exhaustive coverage.

## 7. Next steps (Stage 1 → Stage 2 handoff)
1. Author full-text pass over the 12 core papers (T2/T3/T4 closest works); replace paraphrases with exact quotes and page numbers; resolve the `[to verify]` items.
2. Scopus/WoS counter-search (§6) and recording of results.
3. Freeze patches after the LCZ check; run small CPU-only ingestion for one patch (`--layer weather` first) once you approve downloads.
4. Strategist Phase 3 (outline + 7-dimension reviewer assessment) → `academic-paper` Stage 2.
