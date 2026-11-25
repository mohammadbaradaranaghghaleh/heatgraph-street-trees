# Simulated peer review — *Landscape and Urban Planning*

**Manuscript:** Physics-informed graph surrogates for exposure-weighted street-tree allocation: generalisation to unseen cities and independent verification with SOLWEIG
**Source reviewed:** `manuscript/main.tex` (flattened `main_flat.tex`), tables T1–T6, figures F1–F7 (PNG copies from `figures/`)
**Date:** 2026-11-25 · **Skill:** academic-paper-reviewer v1.11.1, `full` mode, strict stance

> **How this panel was run (read before relying on it).** All five seats and the synthesis were written in one Claude session that had read the whole manuscript and the project code. They are role-separated perspectives, **not independent reviewers**: same model, same context, seats could see each other's reasoning. The skill's helper scripts (sprint-contract checker, acronym check, suggested-reference check, reviewer-claim check) are not installed in this skills copy and **did not run**. Status: `NOT_CALIBRATED`. Three claims were checked against code: NSGA-III seeding (`scripts/08_benchmark.py:116-120`), the "affected segment-hours" definition (`scripts/09_solweig_verify.py:128`), and how the conformal half-width enters the objective (`src/heatgraph/optimize/additive.py:43-47`).

---

## Editorial decision: **MAJOR REVISION** · Readiness **60 / 100**

| Dimension | Judgement |
|---|---|
| Technical soundness of the engine and surrogate | Mostly meets |
| Validity of the headline decision claims (optimisation, "conservative", equity) | Partly meets |
| Fit to LUP readership / planning relevance | Partly meets |
| Literature and positioning | Mostly meets |
| Transparency and reproducibility | Partly meets (code "on publication") |
| Writing and presentation | Meets |

Several things would hold up well in review. The validation design is honest: whole held-out patches, a test set locked before modelling, and a no-change skill reference. The paper reports its negative results openly (greedy matches NSGA-III, linear comes close to the GNN, GATv2 fails). Every reported outcome is re-simulated, and the verification uses an external model. Most papers in this space do none of these. The weaknesses lie elsewhere. The *decision* claims go beyond what the design shows, and the planning contribution LUP readers need is underdeveloped. Neither problem is fatal, but together they could cause a desk rejection or a reject-and-resubmit if left unaddressed.

---

## Reviewer reports (condensed)

### Card #1 — Journal-fit reviewer (LUP Associate Editor)
- **Fit risk (Major).** The title, abstract and contribution list are written for an ML/urban-climate methods audience (GNN, conformal, NSGA-III, v1/v2 edges). LUP publishes planning and design knowledge. As framed, the editor could redirect it to *Urban Climate*, *Building and Environment* or *Sustainable Cities and Society*. [title, abstract, §1 contributions]
- **Practical magnitude (Major).** The best layouts remove 1.2–7.3% of person-weighted heat dose at 5–40% of slots (Table 4). The paper never says whether that is meaningful for a municipality, or what it costs per tree. [Table 4, §5.2]
- **Missing submission items.** No Highlights, no graphical abstract in the source. The code and data are promised only "on publication" [Data availability]. Check the current LUP Guide for Authors on abstract length and required items.
- **Strength.** A clear "decision reliability" narrative, plus explicit qualifying results in §1.

### Card #2 — Peer Reviewer 1 (urban-climate modelling / SOLWEIG)
1. **NSGA-III vs greedy is partly circular (Major).** Both NSGA-III runs are seeded with the linear-surrogate greedy path, and the archive keeps every seed. So "NSGA-III matched greedy" (§4.4) is largely guaranteed by construction (§3.8 admits it weakly dominates the seeds). There is also no unseeded NSGA-III run and no hypervolume comparison, so the reader cannot tell whether the search contributes anything. [§3.8; `scripts/08_benchmark.py:116-120`]
2. **The "conservative" claim is selective (Major).** For dose, the engine and GNN under-predict SOLWEIG by 10–21%. Table 6, however, shows mean ΔUTCI on affected segment-hours of −1.86 to −1.91 °C (engine) against −0.63 to −0.80 °C (SOLWEIG). Peak cooling is −7.25 vs −6.07 °C, and Fig. 7b lies below the 1:1 line at strong cooling. So the chain *over*-predicts cooling under the new crowns and *under*-predicts its spatial spread. The abstract and §5.3 still say "conservative" without qualification. "Affected" uses each model's own threshold (dU < −0.05), so the Table 6 means are not comparable across models. [Table 6, Fig. 7b, §5.3, abstract]
3. **No observational validation (Major for LUP).** The engine is validated against SOLWEIG, and the layouts are "independently verified" by SOLWEIG. Both checks are model against model. Tmrt/UTCI field measurements (e.g., Rotterdam/Utrecht campaigns, or any published Dutch Tmrt dataset) would anchor the absolute magnitudes. If none can be obtained, the wording should be "cross-model verification". [§3.3, §3.9, §4.6]
4. **The conformal "robustness" does little (Major).** The half-width q_h is one constant per hour, added to every segment and every layout. It therefore only shifts the 26 °C threshold to 26 − q_h; it does not penalise uncertain layouts. In addition, the calibration patches are the same ones used for early stopping and model selection. That weakens the exchangeability argument. Coverage was also never reported on the test patches. [§3.6, §3.7; `additive.py:43-47`]
5. **The Freiburg "lower-quality input data" test is mis-described (Major).** The engine labels for Freiburg are computed on the same imputed heights the GNN sees. The test therefore measures transfer to a different height *distribution*, not robustness to input *error* against the true scene. A real test would compare against labels generated on true heights (e.g., LoD2 for Baden-Württemberg). [§3.1, §4.2, abstract]
6. **No sensitivity analysis (Major).** Results rest on fixed choices: crown transmissivity 3%, a single crown geometry (r = 3 m, h = 8 m), wind reduction 0.5, β = 0.002 m⁻¹, an 800 m radius, the 26 °C threshold, and v = 1.3 m s⁻¹. None is varied. At minimum, show how the selected layouts and dose reductions change under ±range for transmissivity, crown size, β and threshold. [§3.2, §3.4, §3.7]
7. **No uncertainty on the results (Minor-to-Major).** All results come from a single training seed and a single NSGA-III seed, with no CIs on the MAEs or dose reductions. The 7% GNN vs linear gap in Rotterdam needs a bootstrap CI or a multi-seed spread. [Tables 3–4]
8. **Calibration rests on one patch-day (Minor).** The ground albedo sits at the edge of its search range (§4.1). Report the range and widen it, or explain why.

### Card #3 — Peer Reviewer 2 (urban forestry / street-tree placement)
1. **The planting model is unrealistic (Major for LUP).** Every slot sits 4 m from the street line every 12 m. There are no constraints for underground utilities, sidewalk width, sight lines, façades or species, and every tree is the same mature 8 m crown planted instantly. A planner would ask what the result looks like at year 5 vs year 30, and how many of these slots are actually plantable. Add at least a crown-size/maturity scenario and a plantability filter discussion. [§3.4, §5.4]
2. **Benchmarks miss the practitioner heuristic (Major).** Random layouts are a weak baseline. The relevant comparison for LUP is what planners actually do: plant the sunniest high-flow streets, rank by shade deficit, or fill canopy gaps. If a simple "unshaded × flow" ranking matches greedy, that is an important (and publishable) finding. [§3.8, Table 4]
3. **Positioning (Minor).** The paper is well cited on the surrogate side. It should engage more with tree-placement and shade-planning literature in planning journals (e.g., shade-equity planning, Tmrt-based tree-planting guidelines, municipal canopy strategy studies), and say explicitly what it adds over Schrodi et al. 2023 beyond the exposure weighting. [§2]
4. **Transpiration and air-temperature effects are left out (Minor).** They are acknowledged, but the conclusion that "most benefit comes from which streets" should be explicitly limited to radiative shading. [§5.4]

### Card #4 — Peer Reviewer 3 (heat equity / environmental justice planning)
1. **The equity metric doesn't match its population (Major).** Dose accrues to *pedestrians on segments* (gravity flows), while "per-capita dose" divides by *floor area in districts* (residents + workers). The two populations differ, so the Theil index has no clear referent. State whose exposure is being equalised. [§3.7, Table 5]
2. **Four arbitrary districts (Major).** It is unclear how the districts are drawn (quadrants?). With four units, the Theil is very sensitive to boundaries (modifiable areal unit problem, MAUP). Theil values are also small (0.03–0.11), so "8.2% less inequality" means 0.0338 → 0.0311. Report absolute district doses, and test sensitivity to the zoning.
3. **Levelling down (Major).** A lower Theil at 90% cooling may come from cooling the advantaged district less rather than the disadvantaged one more. Show per-district dose changes for greedy vs the equity layout.
4. **Flows (Major).** Uniform trip generation without calibration against counts drives which streets win (§5.4 admits this). Run at least one alternative flow assumption (residential origins, transit-weighted destinations) and report how much the layouts overlap (e.g., Jaccard index of selected slots).
5. **Practice pathway (Minor).** How would a municipality use this? Inputs, runtime, who runs it, what the output map looks like. The repo has a dashboard (`dashboard/`, `app.py`) that the paper never mentions; it could support a short "use in planning practice" subsection.

### Fixed seat — Devil's Advocate ("Reviewer #2")
**Strongest counter-argument.** The paper's own results show that the GNN, the conformal layer and NSGA-III do not change the decision. NSGA-III on the linear surrogate gives the same layouts and the same engine-verified dose reductions as NSGA-III on the GNN (Table 4, max difference 0.05 pp). Greedy on the linear model matches both. The conformal bound is a uniform threshold shift. Once crown-shadow features exist, a linear model plus greedy, essentially a shade-deficit-times-flow ranking, delivers every cooling result in the paper. The only thing the heavy machinery adds is the equity front: a 3–8% relative change in a Theil index built on a population proxy the authors themselves say cannot support distributional conclusions. Stated bluntly: the decision-relevant contribution is "physics-informed shading features + greedy", and the GNN/conformal/NSGA-III layers are not shown to be needed.

| # | Severity | Issue | Anchor |
|---|---|---|---|
| DA-1 | **CRITICAL** | The core claim that the GNN surrogate enables better allocation has no decision-level evidence: the linear and GNN optimisers give identical engine-verified outcomes. | Table 4; §4.4; abstract |
| DA-2 | MAJOR | "Conservative, independently verified" cooling conflicts with the engine/GNN over-predicting per-segment cooling. | Table 6; Fig. 7b; abstract; §5.3 |
| DA-3 | MAJOR | "Decision-grade" relies on a conformal bound that is a uniform shift and on model-vs-model verification. | §5.3 |
| DA-4 | MAJOR | Equity conclusions rest on a pedestrian/floor-area mismatch and four undefined districts. Unsupported premise: that equalising district-level pedestrian dose benefits the vulnerable. | §3.7; Table 5; §5.2 |
| DA-5 | MAJOR | NSGA-III ≈ greedy is guaranteed by seeding, so "This result sets a benchmark for the field" is an overclaim. | §3.8; §5.2 |
| DA-6 | MINOR | Generalisation is claimed from three Rotterdam patches plus one Freiburg patch; Rotterdam patches share forcing and data source. | §3.5; T1 |

**Ignored alternative.** A rule-based heuristic (unshaded fraction × flow) was never tested.
**Missing stakeholders.** Residents vs commuters; maintenance and utility owners; species and nursery supply.
**"So what?" test.** It passes only if the paper reframes itself around what planners learn: where trees help, by how much, and at what equity cost. As written, the framing is "our surrogate is accurate".

**DA-1 adjudication (editor).** Validated. It does not lead to rejection, because the paper already reports the linear result honestly and the GNN's value at high density and under input shift is real (Fig. 4). It does, however, require reframing the contributions and the abstract. Accept is blocked until this is addressed.

---

## Revision roadmap (ordered by rejection risk)

| # | Required change | Addresses |
|---|---|---|
| R1 | Reframe for LUP. Lead with planning findings (where trees help, practitioner-baseline comparison, equity–cooling trade-off). Present the GNN as an enabling tool, and state plainly that the linear physics model + greedy suffices for the cooling objective. Revise title, abstract, contributions and §5.2. | Fit, DA-1, DA-5 |
| R2 | Add a practitioner-heuristic baseline (e.g., rank slots by unshaded receptor share × flow; canopy-gap filling) to Table 4/Fig. 5. | R2-2, DA alternative |
| R3 | Run unseeded NSGA-III (and/or report hypervolume for seeded vs unseeded). Drop "sets a benchmark for the field". | R1-1, DA-5 |
| R4 | Qualify "conservative". Report per-segment ΔUTCI bias and MAE vs SOLWEIG with a common "affected" mask. State that the chain over-predicts local cooling and under-predicts its spread. Change "independent verification" to "cross-model verification" unless field data are added. | R1-2, R1-3, DA-2 |
| R5 | Either make the uncertainty term layout-dependent (per-segment or per-layout conformal, or locally weighted) or describe it honestly as a threshold margin. Calibrate on a split not used for early stopping, and report empirical coverage on the test patches and the optimised layouts. | R1-4, DA-3 |
| R6 | Sensitivity analysis: crown transmissivity, crown size/maturity, β/radius, the 26 °C threshold, wind factor. Report changes in dose reduction and selected-slot overlap. | R1-6, R2-1, R3-4 |
| R7 | Equity redesign: define the districts, test alternative zonings, report absolute per-district doses (greedy vs equity layout) to rule out levelling down, and state whose exposure the metric equalises. Ideally use CBS (Statistics Netherlands) 100 m grid population and age data, which are open, in place of the floor-area proxy. | R3-1…3, DA-4 |
| R8 | Uncertainty: multi-seed training (≥3) and bootstrap CIs for Table 3; CIs over random layouts in Table 4. | R1-7 |
| R9 | Fix the Freiburg framing (distribution shift, not input error), or add true-height labels as a real input-error test. | R1-5 |
| R10 | Make code and data available to reviewers now (anonymised GitHub/Zenodo link), and add Highlights and a graphical abstract. | Fit |

### Status after revision round 1 (branch `revision/lup-r1-r4`)
- **R1 done.** New title, abstract, keywords, contributions, §3.8 heuristics, §5.3 "Implications for tree-planting practice", and conclusions.
- **R2 done.** The sun × flow practitioner rule reaches 46–85% of greedy's engine-verified reduction; sun only reaches 29–71% (Table 4, Fig. 5).
- **R3 done.** Unseeded NSGA-III reaches 60–91% of greedy; its front hypervolume is 2–6% lower than the seeded front. "Sets a benchmark for the field" is removed.
- **R4 done, and it corrects this review.** On a common mask, engine and GNN mean cooling are 6–16% *smaller* than SOLWEIG's, so "conservative" holds in total. The earlier Table 6 gap (−1.9 vs −0.7 °C) came from each model using its own mask. Under the crowns the engine over-predicts strong cooling by 0.12–0.39 °C, and it cools only 32–36% as many segment-hours. DA-2 is therefore downgraded from "conflict" to "local over-prediction, now stated".
- Scripts: `scripts/12_revision_experiments.py`, plus updates to `10_figures_tables.py`, `build_facts.py`, `check_numbers.py` and `reproduce.py`. `reproduce.py` passes, with 75 audited claims.

### Status after revision round 2
- **R5 done.** The margin is described as an hourly threshold calibration. Empirical coverage is 86–92% per patch on test plantings (90% pooled) but 79–89% on optimised layouts, which the paper reports.
- **R6 done.** Table 7 covers transmissivity, crown size, wind factor, β and threshold. Every method comparison holds; crown size dominates the magnitude.
- **R7 done, against the hoped-for result.** Districts are defined (core quadrants) and per-district reductions are reported. Equity gains come mainly from cooling the least exposed district less (19 of 20 cases), not from cooling the most exposed more.
- **R8 done.** Over 5 NSGA-III seeds, cooling has SD ≤ 0.02 pp; equity gains vary strongly between seeds (SD up to 6.3 pp) and are framed as examples. Training-seed variation of the GNN is not assessed (stated in limitations).
- **R9 done.** Freiburg is framed as a cross-country, input-regime distribution shift.
- **R10 / minor done.** v1/v2 renamed; abstract has 3 acronyms and 250 words; affected segment-hours defined; no overfull boxes; licences, access date and the GitHub URL added; Highlights and cover letter created.
- **Still open.** Census population data (R7 proxy), field measurements, plantability constraints.

## Minor issues
- The "v1/v2" labels are internal jargon. Name them (e.g., "geometric edges" vs "crown-shadow edges"). [§3.5, Table 3]
- The abstract has about 10 acronyms and many numbers. Cut to the 3–4 planning-relevant results. [abstract]
- Table 6: "affected segment-hours" is undefined in the text, and each model uses its own mask. [Table 6 caption]
- Fig. 6b: the colourbar "mean −ΔUTCI" reaches 3 °C while Table 6 means are about 0.7 °C. State the averaging (per segment over hours, all vs affected). [Fig. 6]
- Fig. 5 caption: "share of the 250 candidate slots". Confirm that every test patch has exactly 250 slots; §3.4 gives 144–250 per patch. [Fig. 5]
- Fig. 4: the x-axis runs to 50% but the data stop at about 35%. Bin centres differ from the stated 5/15/25/35. [Fig. 4]
- "Decision-grade" (§5.3) and "provably near-optimal" both need hedging. Submodularity of the dose objective with the max(·, 0) threshold was not shown.
- §3.7: why 08:00–19:00 only? Evening exposure in canyons may matter.
- §4.1: report the calibration search ranges, since the albedo sits at an edge.
- Table 1: add the number of candidate slots per patch.
- Citations dated 2026 (kamath2026, osm2026, overture2026): make sure they have version and access dates.
- Data statement: Overture/OSM licences (ODbL) should be named.

## Advisory checks
- Acronym check, suggested-reference check, reviewer-claim check: **did not run** (scripts not present in this skill install).
