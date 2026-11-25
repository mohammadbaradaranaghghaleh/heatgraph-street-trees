# Final panel re-review — *Landscape and Urban Planning*

**Manuscript:** Where should street trees go? Exposure-weighted allocation with physics-informed surrogates, tested against practitioner rules and verified with SOLWEIG
**Reviewed:** `manuscript/main.tex` (commit `8b42a25`), rendered `main.pdf` (27 pages), `highlights.txt`, `cover_letter.tex`
**Baseline:** `review/LUP_panel_review_2026-11-25.md` (round 1: Major Revision, 60/100)
**Date:** 2026-11-25 · **Skill:** academic-paper-reviewer v1.11.1, `re-review` mode (same five seats and Round-1 cards)

> **How this was run — read before relying on it.** As in round 1, all five seats and the synthesis were written in one Claude session. That session also wrote the revision, so this is a self-review by role-separated perspectives, not an independent one. The skill's contract checkers and advisory scripts are not installed and did not run. Status: `NOT_CALIBRATED`. **The journal's Guide for Authors could not be read:** ScienceDirect refused the automated fetch (HTTP 403) and showed a CAPTCHA in the browser. The guideline checks below therefore use the criteria you specified (abstract ≤ 250 words, 3–5 highlights ≤ 85 characters, 3–6 keywords, APA 7). Every guideline item is marked **verify against the live guide**. Mechanical checks were measured, not estimated: abstract and keyword counts come from the rendered PDF, highlight lengths in characters, and the reference style from the PDF's reference list.

---

## Editorial decision: **MINOR REVISION** (ready to submit after the fixes in §4) · Readiness **82 / 100**

| Dimension | Round 1 | Final |
|---|---|---|
| Technical soundness (engine, surrogate) | Mostly meets | Meets |
| Validity of decision claims | Partly meets | Mostly meets (one factual error, one overclaim) |
| Fit to LUP / planning relevance | Partly meets | Meets |
| Literature and positioning | Mostly meets | Mostly meets |
| Transparency and reproducibility | Partly meets | Mostly meets (code public; results archive still "on publication") |
| Writing and presentation | Meets | Mostly meets (format items in §2) |

**Why the score is 82, not higher.** The scientific substance of every round-1 item has been addressed, mostly convincingly. Three kinds of problem remain:
- (a) Small but **must-fix** errors that an editor or a hostile reviewer would catch at once: a misstated percentage, a rendered abstract over 250 words, 7 keywords, and a methods count that no longer matches its list.
- (b) Two places where the wording still runs ahead of the evidence: "cooling optimum", and the role claimed for the GNN in equity work.
- (c) Residual limits that are stated honestly but that a strict LUP reviewer may still ask to reduce: one SOLWEIG patch, no field data, a population proxy, and a single GNN training run.

None of (a) or (b) needs new computation beyond about an hour of text edits. The items in (c) are why this is not Accept-as-is from a strict reviewer.

**Why the score is not lower.** The round-1 critical issue (DA-1: "the GNN does not change any decision") is now *conceded and reframed* rather than hidden. The paper states plainly that linear features + greedy reach the cooling result. It shows that a practitioner rule captures only 46–85% of the reduction, and it reports the negative equity and seed findings in full. For a Reviewer #2, that honesty is the strongest asset of the revision.

---

## 1. Traceability matrix: round-1 weaknesses (R1–R10, DA issues)

| Item | Round-1 concern | What the manuscript now does | Anchor | Verdict |
|---|---|---|---|---|
| **R1 / DA-1** GNN vs linear framing | Heavy ML did not change decisions; framing was ML-first | Title, abstract, contributions and §5.3 lead with the planning question. Contribution 2 and §5.3 say linear crown-shadow features + greedy reach the cooling result, and that the GNN helps mainly where crowns overlap. | Abstract; §1 list; §5.3 ¶2 | **Resolved**, with one residual. §5.3 recommends "a fuller surrogate … where equity trade-offs are needed", but the equity front was only computed with the GNN. No linear-NSGA equity front was compared, so that recommendation is untested (see §3, M2). |
| **R2** Practitioner baseline | Only random baselines | Two heuristics (sun × flow, sun only), defined in §3.8, in Table 4 and Fig. 5. 46–85% and 29–71% of greedy; 1.4–3.3× random. | §3.8; §4.4 ¶2; Table 4 | **Resolved.** Residual (minor): one author-designed rule stands for "practice"; a reviewer could call it a straw man (see §3, M3). |
| **R3** Unseeded NSGA-III | Match with greedy built in by seeding | Unseeded run reaches 60–91% of greedy; hypervolume 2–6% lower; "sets a benchmark" removed. | §3.8 ¶3; §4.4 ¶3; §5.2 | **Resolved.** |
| **R4 / DA-2** Common-mask SOLWEIG | Means not comparable; "conservative" selective | Common mask defined (§3.10); engine mean 6–16% below SOLWEIG; local over-prediction under crowns 0.12–0.39 °C; spread 32–36%; §5.4 says single streets are an upper estimate and totals a lower one. | §3.10; §4.7 ¶2; Table 6; §5.4 | **Resolved, but a factual error was introduced/kept in §5.4:** "found 10–20% *more* cooling than the engine". 10–20% is how far the engine *underestimates* relative to SOLWEIG; SOLWEIG's cooling is **11–25% more** than the engine's (2.40/2.16 = 1.11 … 5.92/4.75 = 1.25). Must fix (F1). |
| **R5 / DA-3** Conformal margin | Uniform shift sold as robustness | §3.6 now calls it an hourly threshold-calibration margin. Coverage: 86–92% on test plantings vs 79–89% on optimised layouts (as low as 72%). §5.4 says re-simulation, not the margin, underpins results. | §3.6; §4.2 ¶3; §5.4 | **Resolved.** Exemplary candour. Calibration still reuses the early-stopping patches; this is now harmless because the margin is not used for claims. |
| **R6** Sensitivity | No sensitivity analysis | §3.9 and Table 7: transmissivity, crown size, wind factor, β, threshold. Method comparisons hold everywhere; crown size dominates the magnitude; Jaccard 0.71–1.00 under objective changes. | §3.9; §4.6; Table 7; §5.3 | **Resolved.** Physical settings are not re-optimised, which is stated in Table 7's caption. Acceptable. |
| **R7 / DA-4** Equity honesty | Proxy mismatch, undefined districts, levelling down | Districts defined (core quadrants). Per-district analysis: the lowest-dose district gets less cooling in 19/20 cases, the highest gains median 0.15 pp. The abstract says "mainly by cooling the least exposed district less". Limitations cover the pedestrian-vs-resident mismatch and quadrant dependence. | §3.7; §4.5 ¶2; §5.3 ¶3; §5.5 | **Resolved on honesty.** Not resolved on substance: census data were not used (stated as future work). Acceptable for submission, but a likely reviewer request. |
| **R8** Multi-seed stability | Single seeds, no CIs | 5 NSGA-III seeds: cooling SD ≤ 0.02 pp; equity unstable (SD up to 6.3 pp; 3/100 combinations worse than greedy); Table 5 framed as examples. | §4.5 ¶3; §5.5 | **Partly resolved.** The optimiser seed is covered, but the bootstrap CIs on Table 3 (R1-7) and GNN training-seed variation were **not done**. The "7% / 16% more accurate" and density-bin claims (−1 to 11%) therefore have no uncertainty (see §3, M4). |
| **R9** Freiburg framing | Mis-described as input-error test | §3.1 explains that labels use the same imputed heights, so the test measures distribution shift, not height error. Abstract and conclusion reworded. | §3.1 ¶3; abstract; §6 | **Resolved.** |
| **R10** Availability, Highlights | Code/data on publication; no Highlights | Code public (GitHub URL); licences named; Highlights and cover letter exist. | Data statement | **Partly resolved.** The processed results (labels, trained models, benchmark outputs) are still "Zenodo on publication; to reviewers on request". A strict reviewer wants a reviewer-accessible link now. |
| DA-5 "benchmark for the field" | Overclaim | Removed; replaced by a suggestion to report greedy plus an unseeded run. | §5.2 | **Resolved.** |
| DA-6 Generalisation from 4 patches | 3 Rotterdam + 1 Freiburg | Unchanged design; limitations note the Dutch climate. | §5.5 | **Open (acknowledged).** No action possible without new data. |

**Summary:** 9 resolved (one of them with a must-fix error), 2 partly resolved (R8, R10), and 1 open but acknowledged (DA-6).

---

## 2. Compliance with the journal checklist

| Item | Requirement used | Measured | Status |
|---|---|---|---|
| Abstract length | ≤ 250 words | **255 words as rendered** (250 in the LaTeX source; forms such as "1.30–1.62 K" and "0.063 °C" split into extra words) | **Fail — trim about 10 words** (verify how the guide counts) |
| Highlights | 3–5 bullets, ≤ 85 characters | 5 bullets; 62, 80, 81, 71, 70 characters | Pass. Bullet 4 ("reached the cooling optimum") repeats the M1 overclaim. |
| Keywords | 3–6 | **7** (Urban heat; Street trees; Tree planting strategy; Pedestrian heat exposure; Thermal comfort; Graph neural networks; Multi-objective optimisation) | **Fail — drop one** (verify the guide's limit) |
| Reference style | APA 7th | `elsarticle-harv` output (Elsevier Harvard: "Author, A., Year. Title. Journal Vol, pages. doi:"). Not APA 7: no parentheses around the year, no issue numbers, DOIs not as URLs. One entry duplicates the arXiv ID (Angelopoulos & Bates, 2021). | **Fail against APA 7.** Many Elsevier journals accept any consistent style at first submission ("Your Paper Your Way"); **verify**. Fix the duplicate either way. |
| In-text citations | Author–year | natbib author–year | Pass |
| Planning / management relevance | Explicit | §5.3 "Implications for tree-planting practice"; practitioner comparison; plantability and crown-growth caveats | Pass. This is the revision's main gain. |
| Title page | Corresponding author contact | **No e-mail (`\ead`) and no ORCIDs** printed; "Corresponding author." only | **Fail.** Add the e-mail (Elsevier requires it on the title page). |
| Declarations | CRediT, competing interests, funding, data, generative AI | All present | Pass |
| Graphical abstract | Optional or required? | None | **Verify** against the guide |
| Review model | Single- vs double-anonymised | Manuscript shows names and affiliations | **Verify.** If LUP is double-anonymised, a separate title page and an anonymised manuscript are needed. |
| Length | Research-paper limit | About 9,000 words of body text, 7 figures, 7 tables plus 1 appendix table | **Verify** against the word limit |
| Cover letter | Completeness | Article type, question, findings, honest limits, originality, competing interests, data/code link, AI declaration | Mostly complete. **Missing:** corresponding-author e-mail and phone; the "Research Paper" article-type name should be checked against the guide. Optional: suggested reviewers, note of no related submissions. |

---

## 3. Remaining scientific and wording issues (Reviewer #2 view)

**M1 — "cooling optimum" overclaim (fix before submission).** Contribution 2 says greedy "reaches the cooling optimum that a multi-objective search also finds". §5.3 says greedy "reached the cooling optimum", and Highlight 4 says it "reached the cooling optimum". The optimum is unknown. What was shown is that no search found more cooling (seeded NSGA-III within −0.05 to +0.02 pp; unseeded 60–91%). Write "matched the best cooling that any tested search found".

**M2 — the GNN's niche is asserted, not shown at decision level.** §5.3 recommends "a fuller surrogate where planting is dense or where equity trade-offs are needed". The density advantage is shown in prediction error (Fig. 4) but not in decisions (Table 4: NSGA-linear = NSGA-GNN at every budget). The equity front was computed only with the GNN. Either add a statement that this is untested, or soften to "where prediction accuracy under dense planting matters".

**M3 — one practitioner rule.** The sun × flow rule is reasonable and clearly defined, but it is the authors' construction. A reviewer may propose other rules, such as UTCI hot-spot maps from a standard SOLWEIG run or municipal canopy-gap targets. Add one sentence to the limitations: the comparison covers one plausible rule family, and other practice rules were not tested.

**M4 — no uncertainty on the surrogate comparison.** "7% / 16% more accurate" and the density bins (−1, 7, 10, 11%) come from one trained model, without CIs. A cheap fix is a bootstrap over plantings for Table 3 and Fig. 4 (CPU, minutes). At minimum, state that these differences are from a single training run.

**M5 — "Where trees go matters as much as how many are planted"** (abstract, conclusion, Highlight 1). This is supported against *random* placement ("a quarter to a half as many well-placed trees"). Against the realistic alternative, the practitioner rule, the placement gain is 15–54%. The sentence is rhetorical but defensible; "can matter as much as" is safer.

**M6 — residual design limits** (acknowledged; likely reviewer requests): verification on one SOLWEIG patch, no field Tmrt, a floor-area population proxy, one crown geometry in the main analysis, and three heatwave days. They are stated, so no action is needed for submission. Expect them in reviews.

---

## 4. Must-fix list before upload (in order)

| # | Fix | Where | Effort |
|---|---|---|---|
| F1 | "found 10–20% **more** cooling" → "found 11–25% more cooling", or keep "underestimated by 10–20%" phrasing only. Update the anchor in `check_numbers.py`, which currently enforces the wrong sentence. | §5.4 ¶1 | 5 min |
| F2 | Trim the abstract to ≤ 250 **rendered** words (e.g., drop "on held-out days", shorten the SOLWEIG sentence). | Abstract | 10 min |
| F3 | Reduce keywords to 6 (suggest dropping "Thermal comfort" or "Multi-objective optimisation"). | Keywords | 1 min |
| F4 | "cooling optimum" → "the best cooling any tested search found" (M1). | §1 contribution 2; §5.3 ¶2; Highlight 4 | 5 min |
| F5 | "We compared four methods" → "We compared the following methods" (the list has five, plus the unseeded run). | §3.8 | 1 min |
| F6 | "while retaining 90–100% of greedy's cooling" → "90–94%" (that is the range at 50–100 trees in Table 5; 100% occurs only at 25 trees). | §4.5 ¶1 | 2 min |
| F7 | Name the layout for the peak-cooling comparison ("for the high-budget layout, its strongest…"). | §4.7 ¶2 | 1 min |
| F8 | Add the corresponding-author e-mail (`\ead{}`) and, ideally, ORCIDs. | Title page; cover letter | 5 min |
| F9 | Remove the duplicate arXiv ID in the Angelopoulos & Bates entry; decide on APA 7 vs Elsevier Harvard after checking the guide (if APA is required, switch to an APA `.bst`/`biblatex-apa`). | `references.bib` / style | 10–60 min |
| F10 | Add the M3 limitation sentence and the M4 single-run caveat (or a bootstrap). | §5.5 | 10 min |
| F11 | Soften the §5.3 "fuller surrogate … equity" recommendation (M2). | §5.3 ¶2 | 5 min |
| F12 | Check the live guide: review anonymity, graphical abstract, word limit, article-type name, keyword limit, reference style. | — | You (the site blocks automated access) |

**Optional (raises the score toward 90):** a reviewer-accessible Zenodo link (private-link sharing) for the results archive (R10); a bootstrap CI for Table 3 (M4); 2–3 GNN training seeds (R8).

---

### Fix status (applied after this review)
- **Applied:** F1 (11–25% more; the audit anchor now uses a dedicated fact), F2 (246 rendered words), F3 (6 keywords), F4 (optimum wording in contribution 2, §5.3, conclusion, Highlight 4, cover letter), F5, F6, F7, F9 (duplicate arXiv IDs removed from all 10 arXiv entries and from `build_bib.py`; style left as Elsevier Harvard pending F12), F10 (new limitation paragraph), F11, and M5 ("can matter as much").
- **Not applied:** F8 needs the corresponding author's e-mail, which was not provided. F12 needs a person to read the guide.
- `check_numbers.py` passes (89 claims); `reproduce.py` passes; both PDFs have no overfull boxes.

## 5. Seat summaries

- **Journal-fit reviewer (EIC card):** Fit is now good. The planning question leads, the practitioner comparison and the practice section are exactly what LUP readers need, and the honest qualifications help. Format items F2, F3, F8 and F12 must be cleared before upload. Recommendation: send to review after minor fixes.
- **Reviewer 1 (methodology):** R3, R4, R5, R6 and R9 are convincingly handled; the common-mask analysis even corrected round 1's own critique. Two items remain: the factual error F1, and no uncertainty on the surrogate comparison (M4). Minor Revision.
- **Reviewer 2 (urban forestry):** The practitioner baseline, the crown-size sensitivity and the plantability caveat answer round 1. A single rule family (M3) and no species or maturity scenario in the main results are acceptable as stated limitations. Minor Revision.
- **Reviewer 3 (heat equity):** The equity reporting is now honest: it names levelling-towards-even through less cooling of the least exposed district, seed instability and quadrant dependence. The substance (census population, district definition) is unchanged, which a strict equity reviewer will still request. Minor-to-Major on substance, Minor on honesty. Net: Minor Revision.
- **Devil's Advocate:** The round-1 strongest counter-argument ("the GNN, conformal and NSGA-III layers do not change decisions") has been **accepted by the authors** and turned into a finding. What remains is (i) the "optimum" overclaim, (ii) the unshown decision-level GNN niche (M2) and (iii) the F1 arithmetic error. No CRITICAL issues remain; DA-1 is adjudicated as resolved by reframing.

## 6. Verdict

**Minor Revision: submit after F1–F11 (about 1–2 hours) and the guide check F12.** Expected outcome at LUP if submitted after those fixes: a single round of minor-to-moderate revision. The most likely requests are census population data, more SOLWEIG patches or field data, and an uncertainty analysis for the surrogate comparison.
