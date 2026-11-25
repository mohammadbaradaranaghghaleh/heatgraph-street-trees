# HeatGraph — Methodology Plan v2 (pure-Python 2.5D engine)

*Status: Stage 3 done (2026-11-25): real data ingested, engine calibrated and accepted against SOLWEIG-GPU (§6). Supersedes Idea 1 in `../../Q1_Env_Urban_Ideas.md`.*

## 0. What changed from v1, and why

| v1 (idea note) | v2 (this plan) | Reason |
|---|---|---|
| SOLWEIG via QGIS/UMEP for labels | **Own pure-Python 2.5D engine** (numpy + pvlib + pythermalcomfort) | No GUI dependency; full control; vectorised; reproducible in CI |
| SOLWEIG as truth | **`solweig-gpu` (pip, GPL-3.0) as an external reference** on 4 patches | Scripted SOLWEIG now exists, so engine fidelity can be benchmarked without QGIS |
| Any city | **17 candidate 500 m patches from 3DBAG (NL)**, LCZ-stratified; Rotterdam held out for full-city evaluation; Freiburg (Overture) for cross-dataset transfer | 3DBAG gives LiDAR-derived heights; Freiburg links to the existing U-Net Tmrt work |
| "First GNN for microclimate" | **Repositioned novelty** (see §1) | Stage 1 search found GNN microclimate predictors (UrbanGraph, ICLR 2026; Yu et al., 2024, *Urban Climate*) — a GNN alone is not new |

## 1. Contribution after repositioning

What is **not** new: neural surrogates of Tmrt (U-Net emulation of SOLWEIG, Briegel et al., 2023), surrogate-plus-search tree placement (arXiv:2310.05691; SU-MHGA 2026 preprint), GNN microclimate prediction (UrbanGraph; Yu et al., 2024).

What **is** claimed (each to be confirmed in Quality Gate 2):
1. **Intervention-faithful surrogate.** The model is trained and tested on *counterfactual edits* (tree insertions re-simulated by the engine), not only on observed scenes. Existing surrogates are scored on held-out scenes, which says little about the edits an optimiser makes.
2. **Person-weighted objective on the pedestrian network.** The optimiser targets heat dose experienced along walked routes (UTCI × estimated pedestrian flow × vulnerability), not pixel-mean Tmrt.
3. **Equity as an objective, not a post-hoc map.** Theil/Gini of dose across neighbourhoods is optimised jointly.
4. **Decision-grade uncertainty.** Conformal intervals inside the objective, plus re-simulation of the final Pareto set (surrogate-optimality gap).
5. **Cross-dataset transfer.** Train on 3DBAG (LiDAR heights) and test on Overture heights (partial/imputed), which quantifies how input-data quality degrades the decision.

## 2. Research questions

- **RQ1** How accurately, and with what calibrated uncertainty, does a graph surrogate reproduce engine UTCI for *unseen edits* and *unseen cities*?
- **RQ2** How much do tree allocations that are optimal for person-weighted dose differ from those optimal for pixel-mean Tmrt (spatial overlap, dose difference)?
- **RQ3** What is the trade-off between total dose reduction and distributional equity at fixed tree budgets?

## 3. Pipeline (Blocks A–I → `src/heatgraph/`)

| Block | Package | Status | Compute |
|---|---|---|---|
| A Ingestion | `ingest/` | **Skeleton implemented** (dry-run default) | CPU, network |
| B Gridding | `engine/surfaces.py` | Interface only | CPU |
| C Engine | `engine/` | Interfaces + equations documented | CPU (numba optional) |
| D Graph | `graph/` | Spec only | CPU |
| E Surrogate | `surrogate/` | Spec only | **GPU — not until your GPU is free** |
| F XAI | `explain/` | Spec only | GPU/CPU |
| G UQ | `uq/` | Spec only | CPU |
| H Optimisation | `optimize/` | Spec only | CPU + surrogate |
| I Validation | `validate/` | Spec only | CPU (solweig-gpu has a CPU fallback) |

### Block C — engine (detail)
Per patch: 900 m × 900 m window (500 m core + 200 m buffer), 1 m grid; per hour of the three heatwave days.

1. **Sun**: `pvlib.solarposition.get_solarposition` (NREL SPA); beam/diffuse from forcing DNI/DHI, `pvlib.irradiance.erbs` fallback.
2. **Shadows**: shifted-max raster casting on DSM and CDSM (Ratti & Richens, 2004); leaf-on crown transmissivity and trunk zone as in SOLWEIG; `trimesh` ray casting only to unit-test the raster method on small scenes.
3. **SVF**: horizon-angle method, 36 azimuths, 200 m radius; SVF_b and SVF_v kept separate; static per patch, cached.
4. **Tmrt**: six-directional short/long-wave fluxes, standing-person angular factors, absorption 0.70/0.97 (Thorsson et al., 2007; SOLWEIG formulation, Lindberg et al., 2008).
5. **UTCI**: `pythermalcomfort` (Bröde et al., 2012 procedure). UTCI's wind input is defined at 10 m, so street ventilation enters as a *10 m-equivalent* speed (forcing wind × urban reduction factor, tuned per LCZ), clipped to 0.5–17 m/s. (Corrected 2026-11-24: v1 of this plan wrongly reduced wind to 1.1 m before UTCI.)

**Implementation status (Stage 2, CPU):** shadow, SVF, Tmrt, UTCI and the hourly driver (`engine/simulate.py`) are implemented and pass 21 synthetic-scene tests. Measured speed on a 900 × 900 m synthetic patch: about 0.26 s per hour for shadows and about 31 s per patch for SVF (36 azimuths, computed once). Known simplifications awaiting Block I calibration: isotropic walls at one temperature, parametric ground warming, lateral fluxes without a per-aspect split, and crowns treated as solid for SVF.
6. **Receptors**: points every 5 m along walkable edges (sidewalk geometry if present in OSM, else a 3 m offset), aggregated per edge (mean, p90, sunlit hours).

**Engine acceptance gate (before any GPU work):** Tmrt MAE ≤ 3 K and bias within ±1 K against `solweig-gpu` on 4 reference patches × 3 days; shadow unit tests pass. If the gate fails, the engine is fixed or the paper reports it as a limitation. The surrogate is never trained on unvalidated labels.

### Training data design (Block C → E)
- 17 candidate patches → freeze 15–20 after LCZ check against the global LCZ map (Demuzere et al., 2022).
- Per patch: the observed scene plus **K counterfactual edits** (K ≈ 50: random tree insertions/removals under the planting constraints). This is what makes RQ1 answerable.
- Splits: leave-one-city-out on NL patches; Rotterdam fully held out; Freiburg (Overture) as a transfer test.

### Blocks E–H — implemented and tested on a synthetic scene (Stage 2, CPU, 2026-11-24)

Toy scene: 80 × 80 m grid city, 12 street segments, 24 planting slots, 3 heatwave hours. Labels come from the Block C engine for 70 random plantings (40 train / 10 val / 10 calibration / 10 test). `scripts/demo_synthetic_pipeline.py` reproduces the numbers below.

| Result | Sum aggregation (`edgemlp`, default) | GATv2 attention (ablation) |
|---|---|---|
| Test MAE, segment UTCI | 0.075 °C | 0.462 °C |
| Correlation with the engine's tree-induced change | 0.992 | 0.683 |
| MAE of the change (zero-change reference: 0.663 °C) | 0.075 °C | 0.462 °C |
| 90 % split-conformal half-width / test coverage | 0.224 °C / 0.908 | 1.484 °C / 0.931 |

**NSGA-III, v1 (random initial population, pop 30, 15 generations, budget 8):** 14 non-dominated designs, minimum dose 315.3 vs 341.7 with no trees. **Greedy reached 311.3**, so plain NSGA-III lost on dose.

**NSGA-III, v2 (Stage 2b, greedy-seeded + archive).** The initial population holds the full greedy path (best 1…B trees), swap mutants of each greedy design (same tree count, one tree moved), the empty design and random feasible designs. Every evaluated design goes into an archive, and the returned front is the non-dominated feasible subset of that archive. This guarantees the front weakly dominates the greedy path: min dose ≤ greedy dose and HV ≥ HV(greedy), by construction rather than by luck, since NSGA-III's own niching can drop front-1 points. Hypervolume against a shared reference point (1.1 × the worst value of each objective):

| Run (same budget, pop 30, 15 gen) | Min dose | Hypervolume | Designs on front |
|---|---|---|---|
| Greedy path | 311.3 | 0.313 | 8 |
| NSGA-III unseeded | 317.1 | 0.331 | — |
| **NSGA-III seeded + archive** | **311.3** | **0.374** | 37 (0–8 trees) |

On the 3-member-ensemble scene in `tests/test_stage2b.py`, seeded NSGA-III also beats greedy on dose (327.6 vs 329.0), with HV 0.10 vs 0.08 for greedy and 0.07–0.08 for unseeded. Run-to-run HV varies slightly through CPU float non-determinism in training. Re-simulating the 4 lowest-dose designs with the engine gives a relative dose gap ≤ 0.2 %, and the conformal upper bound covers the engine dose in 4/4.

**Uncertainty for acquisition (Block G v2).** Plain split conformal has the same width for every design, so it cannot rank designs by uncertainty. A 3-member deep ensemble (`surrogate/ensemble.py`) gives a per-segment spread, and `NormalizedConformal` scales the calibrated quantile by it. On synthetic heteroscedastic data it keeps 0.88–0.92 coverage, and in the noisiest third it beats constant-width intervals.

**Active learning (Block I, `validate/active_learning.py`).** Each round: calibrate on a fixed calibration set → candidate pool = every feasible design the seeded NSGA-III visits → acquire the designs with the widest conformal dose interval → engine re-simulation → record the gap *before* refitting → stop if max |gap| ≤ threshold, else warm-start the ensemble. Tests check that acquired designs are new and wider than the pool median, that the training set grows, that the loop stops at the threshold, and that the gap on acquired designs does not grow after refitting. **Caveat:** acquired designs are model-selected, so they are not exchangeable with the calibration set. Coverage on them is reported, not guaranteed.

**`build_grids` (Block B)** now rasterises footprints with numpy + shapely only (pixel-centre rule, holes respected, max height on overlaps, out-of-window parts clipped, missing heights imputed from neighbours within 50 m). GDAL/rasterio is needed only to resample a canopy GeoTIFF.

**Design decisions made in testing (all belong in the paper):**
1. **Residual on an engine baseline.** The GNN predicts the tree-induced change on top of one engine run of the unplanted scene. Without it, the surrogate was accurate in absolute UTCI (0.57 °C MAE) but no better than "no change" on the change itself. This is the GAP-001 failure mode reproduced in miniature.
2. **Sum, not attention, for slot→segment messages.** Softmax attention averages over planted slots and loses the tree count. The ablation above quantifies the cost.
3. **Integrated Gradients in float64 with a convergence check.** Attributions cancel strongly (total |attr| ≈ 43 vs a net effect ≈ 0.2 °C), so completeness is judged against attribution mass and step refinement.
4. **Caveat.** An additive ridge model is nearly perfect on this toy (crowns never overlap), so the toy is easy. Real patches with overlapping crowns and SVF interactions are the real test, and the ridge model stays as a baseline there.

### Block H — optimisation
NSGA-III (`pymoo`); decision variable: planting slots on edges (binary/integer); objectives: (i) exposure-weighted dose at the conformal upper bound, (ii) Theil index of dose across neighbourhoods, (iii) number of trees. Baselines: random, greedy shade-deficit, pixel-Tmrt optimum (to answer RQ2).

### Block I — validation
1. Engine vs `solweig-gpu` (above).
2. Re-simulate 10–20 Pareto solutions with the engine and report the surrogate-optimality gap.
3. Active-learning loop: add re-simulated scenes and fine-tune while the gap exceeds the threshold (2–3 rounds).
4. Optional field check: low-cost globe-thermometer campaign at 3–5 sites (strengthens the reply to "model of a model").

## 4. Reviewer #2 pre-emption (updated)

| Likely objection | Built-in answer |
|---|---|
| "Your engine is a toy SOLWEIG." | Engine acceptance gate against `solweig-gpu`, reported in full, plus shadow unit tests. |
| "Model of a model; the optimiser exploits surrogate error." | Counterfactual training data, conformal upper-bound objective, Pareto re-simulation gap, active learning. |
| "UrbanGraph already does GNN microclimate." | Different target (decisions under edits, person-weighted dose, equity) and an explicit head-to-head with a raster U-Net on identical labels. |
| "Pedestrian flows are guessed." | Sevtsuk-style flow model with a sensitivity analysis (flow-weighted vs unweighted vs population-weighted) that shows how the allocation changes. |
| "Netherlands only." | Freiburg transfer test with Overture heights, and height completeness reported as a covariate. |

## 5. Hardware plan

Blocks A–D and G–I run on CPU. Block E (and F if on GPU) waits until your GPU is free. **Nothing heavy has been run yet.**

## 6. Stage 3 — real data and SOLWEIG-GPU acceptance (2026-11-25)

### 6.1 Environments
| Environment | Purpose | Key versions |
|---|---|---|
| `HeatGraph/.venv` | engine, surrogate, optimiser, ingestion | torch 2.14.1+cu130 (RTX 5070, sm_120), PyG 2.8.1, rasterio 1.4.4 (GDAL 3.10.3), geopandas 1.2, overturemaps |
| `solweig-ref` (Miniforge, `C:\Users\User\miniforge3`) | **reference model only** | solweig-gpu 2.0.1, GDAL 3.13.3 (conda-forge), torch 2.14.1+cu130, numba 0.68.1 |

solweig-gpu imports `osgeo.gdal` and `numba` without declaring them as pip dependencies, and PyPI has no Windows GDAL wheel, so the reference runs in its own conda env (`scripts/solweig_ref_run.py`, called via `conda run`). The Miniforge installer was checked against the SHA-256 GitHub publishes for release 26.9.4-2 and its NumFOCUS Authenticode signature.

### 6.2 Ingestion (Block A)
All 18 candidate patches ingested, no errors, ~260 MB (`data/raw/ingest_summary.csv`). Changes forced by real data:
- **Walk network from Overture `segment` + `place` (OSM-derived), not live Overpass.** The public Overpass endpoint answered "at capacity" and asks automated users to run their own instance. `ingest/osm_network.py` remains opt-in (`--layer osm`). Cite OSM and Overture in the data statement.
- **Meta CHM tile id verified:** property `tile`, a 9-digit zoom-9 quadkey (56,145 tiles). A canopy writer bug (inherited COG block size) was fixed.
- **Weather now covers the full local day** (previous UTC day included). The manifest now merges layers when a single layer is re-run.
- **Patch QA flags before freezing:** `rot_kop_van_zuid` (72 buildings, 0.7 % canopy, likely mostly river), `dha_beatrixkwartier` (58 % canopy, likely the Haagse Bos), `ams_zuidas` (median height 6.5 m for an intended high-rise LCZ). Freiburg (Overture): 65.8 % of heights missing → imputed, as intended for the transfer test.

### 6.3 Engine vs SOLWEIG-GPU (Block I gate)
Same 1 m grid, same rasters (Building_DSM = DEM + heights, Trees = height above ground, 3 % transmissivity, 25 % trunk zone), same UMEP met file (local time, preceding-hour radiation). Our engine now evaluates the sun at the interval midpoint (`[sun] time_offset_min = -30`), matching SOLWEIG. Metric: Tmrt over all ground pixel-hours of the 500 m core (~3 M per patch-day). Gate: MAE ≤ 3 K **and** |bias| ≤ 1 K. Calibration used one patch-day only (grid search over ground warming, wall warming, ground albedo). The other four were never used for fitting.

| Patch-day | Role | Default MAE / bias | Calibrated MAE / bias | Gate |
|---|---|---|---|---|
| Amsterdam Grachtengordel, 2019-07-25 | calibration | 1.81 / −1.36 K | 1.32 / −0.40 K | (fit) |
| Utrecht binnenstad, 2022-07-19 | held-out | 1.71 / −1.11 K | **1.51 / −0.49 K** | pass |
| Rotterdam Oude Westen, 2020-08-08 | held-out | 1.62 / −0.97 K | **1.30 / −0.26 K** | pass |
| Amsterdam Zuidas, 2019-07-25 | held-out | 2.16 / −1.90 K | **1.43 / −0.73 K** | pass |
| Rotterdam Centrum, 2022-07-19 | held-out | 2.25 / −1.79 K | **1.62 / −0.85 K** | pass |

Calibrated values: ground warming 15 K per kW/m², wall warming 4 K per kW/m², ground albedo 0.30 (now in `engine.toml`). SOLWEIG-GPU takes about 85 s per patch-day on the RTX 5070.

**Residual structure (report in the paper, do not tune away):** daytime bias by stratum after calibration is sunlit −0.04 to +1.87 K, building shade −0.39 to −2.19 K, under canopy +0.07 to +1.23 K. The engine is still too cool in building shade, worst in the high-rise Zuidas. That is consistent with what it lacks compared with SOLWEIG: an anisotropic sky and per-aspect wall fluxes. The fitted ground albedo is at the edge of its range and should be read as an effective parameter that compensates for these omissions. A held-out set with more high-rise patches and the per-aspect wall terms are the next engine improvements.

### 6.4 GPU check
`scripts/gpu_smoke_test.py`: the surrogate trains on CUDA with the same accuracy as on CPU (test MAE 0.096 vs 0.094 °C, 66 MB peak). Bug found and fixed: `HeteroData.to()` is in-place, and the warm-up in `fit()` had moved the caller's first training sample to the GPU. `tests/test_gpu.py` guards against it.

## 7. Stage 4 — frozen dataset, counterfactual labels, surrogate on the GPU (2026-11-25)

### 7.1 Patch re-centring and freeze
Three flagged patches were moved by an LCZ-guided search (`scripts/03_patch_search.py`). It scores candidate centres on a 50 m grid within ±450 m using 3DBAG heights and Overture water, with core metrics from `heatgraph/morphology.py`; every candidate is recorded in `data/cache/search/patch_search.json`.

| Patch | Old core | New core |
|---|---|---|
| `rot_kop_van_zuid` | 51 % water, BSF 0.17, 72 buildings in the window | 15 % water, BSF 0.22, 9 towers > 40 m, h̄ 44.6 m, 497 buildings |
| `dha_beatrixkwartier` | Haagse Bos: BSF 0.05, no towers, 58 % canopy | BSF 0.20, 7 towers, h̄ 42.8 m, 24 % canopy |
| `ams_zuidas` | already high-rise (13 towers, h̄ 37 m); the "median 6.5 m" flag came from small outbuildings | 19 towers, h̄ 45.0 m |

The waterfront rule had to be relaxed (v1: 0/361 feasible windows, because piers have water on several sides). v2 caps water at 35 % with an explicit water penalty; both rules are kept in the script. `ams_bijlmer` (intended LCZ4) has no towers above 40 m; its flats are about 30–35 m. It is kept and noted. The raw dataset is **frozen** in `data/raw/FROZEN.json` (18 patches, 577 files, SHA-256 each). Generation refuses to run if any file has changed.

### 7.2 Counterfactual labels (Block C at scale)
`engine/receptor_engine.py` evaluates the calibrated engine only at pedestrian receptors, batched over plantings in torch on the RTX 5070. Building horizons and shadows are computed once (exact decomposition), and tests show equality with the full-grid numpy engine on CPU and CUDA. Per patch: walk-network segments in the 500 m core, receptors every 5 m, up to 250 planting slots (4 m sidewalk offset, ≥ 8 m apart, not in buildings, water or existing canopy), 81 plantings (index 0 = existing canopy; others with slot density ~ U(0.02, 0.40)), 36 hours (08–19 local, 3 heatwave days). **18 patches, 1,458 plantings, 52,488 (planting, hour) graphs, about 40 s per patch on the GPU, no NaNs.** Mean tree effect per segment −0.09 to −0.23 °C; maximum segment cooling −7.2 to −7.8 °C.

### 7.3 Surrogate training (Block E), split fixed in advance
Train on 14 NL patches (plantings 0–70, 35,784 graphs); validate on the last 10 plantings of the same patches; **test on Rotterdam (3 patches, unseen city)** and **Freiburg (unseen city, Overture heights 66 % imputed)**. Hidden 96, 3 layers, 60 epochs, AdamW + cosine, about 13 min (sum) and 16 min (GATv2) on the RTX 5070.

| Model | Split | UTCI MAE | Change corr. | Change MAE | Zero-change MAE |
|---|---|---|---|---|---|
| **sum aggregation** | val (seen patches, new plantings) | 0.034 °C | 0.987 | 0.034 | 0.197 |
| **sum aggregation** | **Rotterdam (unseen city)** | **0.098 °C** | **0.837** | **0.099** | 0.186 |
| **sum aggregation** | **Freiburg (unseen city + data)** | **0.116 °C** | **0.788** | **0.117** | 0.192 |
| GATv2 (ablation) | Rotterdam | 0.288 °C | 0.164 | 0.291 | 0.186 |
| GATv2 (ablation) | Freiburg | 0.273 °C | 0.167 | 0.277 | 0.192 |

**Findings for the paper.**
1. **The aggregation ablation holds on real data, more strongly than on the toy:** GATv2 is worse than predicting "no change" on both unseen cities, while sum aggregation removes 47 % (Rotterdam) and 39 % (Freiburg) of the change error.
2. **There is a clear generalisation gap:** 0.034 °C on new plantings in seen patches against 0.098–0.116 °C on unseen cities. The best validation epoch was the last one, because within-patch validation rewards memorising patch geometry. Next steps: leave-patch-out validation for model selection, dropout/weight decay, and more patches. This gap is GAP-004 measured directly.
3. **Transfer to Overture heights costs a little more** (0.116 vs 0.098 °C; change corr 0.79 vs 0.84). That is the input-data-quality effect the paper set out to measure.

Artefacts: `data/processed/<patch>/` (scene + labels), `data/models/heatgraph_{edgemlp,gatv2}.pt`, `data/models/metrics_*.json`.

## 8. Closing the seen-vs-unseen-city gap (2026-11-25)

**Diagnosis.** The v1 run validated on held-out *plantings of training patches*. That rewards memorising each patch's geometry: 0.034 °C validation against 0.098 °C on Rotterdam, with the best epoch at the very end.

**Changes (all in the code; ablation isolates each):**
1. *Leave-patch-out validation:* whole patches from the training cities (`ams_bos_en_lommer`, `ein_strijp_s`, `amf_vathorst`, one per morphology family) are used for model selection and early stopping (patience 10). Rotterdam and Freiburg are never used for selection.
2. *Regularisation:* dropout 0.1, weight decay 1e-4.
3. *Physics-informed slot→segment edges (v2 features):*
   - `crown_shade_frac` is the share of a segment's receptors inside the crown's ground shadow that hour and not already in building shadow. The shadow is a "stadium" of crown-radius discs between the trunk-top and crown-top shadow points.
   - `overhead_frac` is the share of receptors under the crown.
   
   They describe the shading mechanism, not a city. On their own they correlate −0.87 (Rotterdam Centrum) and −0.92 (Bijlmer) with the engine's ΔUTCI.

**Result.** Same data and split; change metrics are relative to the engine's unplanted baseline.

| Model | Val (held-out patches) | Rotterdam MAE / corr | Freiburg MAE / corr |
|---|---|---|---|
| v1, within-patch val (old) | 0.034 °C | 0.098 / 0.84 | 0.116 / 0.79 |
| v1 + leave-patch-out val + regularisation | 0.099 °C | 0.100 / 0.83 | 0.116 / 0.79 |
| **v2 physics edges** (`heatgraph_edgemlp_v2_patchval.pt`, default) | **0.063 °C** | **0.062 / 0.91** | **0.074 / 0.88** |
| Linear per-hour model on the v2 features (no GNN) | 0.069 | 0.067 / 0.90 | 0.089 / 0.87 |

- Most of the "gap" was a validation artifact. Under honest validation, v1's validation and unseen-city errors agree (0.099 vs 0.100). Regularisation alone did not help unseen cities.
- The physics edges cut unseen-city error by 38 % (Rotterdam) and 36 % (Freiburg). The change error is now 66 % / 61 % below the zero-change reference (was 46 % / 39 %). Validation and unseen-city errors agree (0.063 vs 0.062): no residual gap.
- **The GNN adds little over the physics feature itself:** +6 % in Rotterdam and +16 % in Freiburg against a linear per-hour model. By planting density (Rotterdam + Freiburg, `scripts/07_density_analysis.py`) the GNN's gain is −1 % below 10 % of slots planted, then +7 %, +10 % and +11 % up to 40 %. That is where overlapping crowns break additivity.
- **Implication for the paper's framing:** the transferable skill comes from the physics-informed graph features; the GNN contributes the interaction (overlap) effects at higher densities and under data-quality shift. For small tree budgets the linear model is a credible, cheaper surrogate and should be reported alongside the GNN, which strengthens rather than weakens the decision-reliability argument (GAP-001).

## 9. Stage 5 — optimisation benchmark, SOLWEIG-GPU verification, figures (2026-11-25)

**Set-up.**
- The v2 GNN is the primary surrogate; the linear physics model (`surrogate/linear.py`, exact fast evaluator in `optimize/additive.py`) is the benchmark surrogate.
- Robust objective: split-conformal upper bound, Mondrian by hour, α = 0.1, calibrated on the leave-patch-out validation patches. q = ±0.077 °C (GNN), ±0.082 °C (linear).
- Objectives: person-weighted dose (UTCI > 26 °C; gravity flows with uniform origins and Overture places as destinations); Theil index across 4 core districts; number of trees.
- **Population is a PROXY** (3DBAG floor area per district) and vulnerability = 1, until CBS/WorldPop is ingested.
- Methods: NSGA-III on GNN (primary), NSGA-III on linear, greedy (linear), random (10 per budget). Budgets: 5–40 % of 250 slots.
- Every chosen design is re-simulated with the calibrated engine. All numbers below are engine-verified.

**Results (unseen test cities; `tables/T4_benchmark`, `T5_equity`, `figures/F5`).**
1. **Optimised layouts cool 2–5× more than random ones at low budgets and about 2× more at 40 %:** 4.7–7.3 % dose reduction against 2.4–3.8 % at 100 trees, consistent across 3 Rotterdam patches and Freiburg.
2. **NSGA-III does not beat greedy on dose.** That held with 15k (GNN) and 50k (linear) evaluations; differences are within ±0.1 percentage points in both directions. Tree shading is close to additive (hence the linear model's strength), and greedy is near-optimal for near-additive coverage objectives. We report this rather than tune it away.
3. **NSGA-III's value is the trade-off greedy cannot see.** At ≥ 90 % of greedy's cooling, the front cuts the between-district Theil by up to 8.2 % (Rotterdam Oude Westen, 75 trees) and by 2.8–5.1 % at mid-to-high budgets elsewhere. The equity lever is modest because trees remove only ~5 % of the dose, and it is bounded by the population proxy.

**Independent SOLWEIG-GPU verification (`rot_centrum`, unseen city; `T6`, `F7`).**
- Five front designs were selected by the GNN objectives, not the engine: none, low budget (25 trees), knee (37), high budget (75), max cooling (100).
- They were run through SOLWEIG-GPU for all 3 heatwave days: 15 runs, about 85 s each on the RTX 5070.
- SOLWEIG confirms the cooling and finds **more** of it: 2.4 / 3.2 / 5.1 / 5.9 % dose reduction against 2.2 / 2.8 / 4.2 / 4.7 % (engine) and 2.1 / 2.8 / 4.1 / 4.7 % (GNN). So the modelling chain is **conservative by 10–20 %**.
- Per segment-hour, ΔUTCI correlates with SOLWEIG at r = 0.95–0.97 (engine) and 0.91–0.94 (GNN).
- SOLWEIG's cooling is spatially broader and less peaked (max −6.0 °C vs −7.3 °C). That is consistent with the shaded-stratum bias found in Stage 3: no anisotropic sky and no per-aspect wall longwave in our engine.

**Figures** (`figures/`, PDF + 300 dpi PNG; validated palette, legends + direct labels):
- F1 engine vs SOLWEIG gate
- F2 surrogate ablation
- F3 GNN vs engine ΔUTCI parity (unseen cities)
- F4 error vs planting density
- F5 benchmark across unseen cities
- F6 rot_centrum Pareto front + knee-design map with SOLWEIG cooling
- F7 SOLWEIG verification

**Tables** (`tables/`, CSV/MD/LaTeX): T1 patches, T2 engine calibration, T3 surrogates, T4 benchmark, T5 equity, T6 SOLWEIG verification. Regenerate with `scripts/10_figures_tables.py`.
