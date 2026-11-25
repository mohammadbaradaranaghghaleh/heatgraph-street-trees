# Pre-upload audit of the submission package — *Landscape and Urban Planning*

**Package:** `main.tex` / `main.pdf`, `manuscript_blind.pdf`, `title_page.pdf`, `highlights.txt`, `cover_letter.pdf` (commit `0d24668`)
**Date:** 2026-11-25 · **Skill:** academic-paper-reviewer v1.11.1, strictest Reviewer #2 stance, re-review of the final package
**Method.** Every check below was **measured**:
- All four documents were rebuilt fresh with full logs. The committed PDFs have text identical to the fresh builds.
- `pdftotext` extracts of the committed PDFs were searched with scripts.
- The blind PDF was additionally scanned in its metadata, link targets and all 198 compressed streams.
- Facts were compared against `facts.json`.

As before, this is a self-audit by the session that wrote the revision, not an independent review. The journal's live Guide for Authors could not be read (ScienceDirect returned 403 and a CAPTCHA), so items that depend on it are marked **unverified**.

---

## Verdict

**Readiness: 91 / 100. Ready to upload after four micro-fixes (≈ 10 minutes) and your check of the live guide (F12).** Scientifically, every round-1 and final-review item is resolved or honestly disclosed. Expected first decision at LUP: **Minor Revision**. Likely requests are census population data, more SOLWEIG patches or field data, and uncertainty on the surrogate comparison, all already listed as limitations.

---

## 1. R1–R10 and F1–F11: consistency across the package

| Claim | main.tex / main.pdf | Blind PDF | Highlights | Cover letter | Status |
|---|---|---|---|---|---|
| SOLWEIG found **11–25%** more cooling (§5.4) | ✓ (facts 11.3–24.7) | ✓ | — | — | **OK** |
| Engine/GNN underestimate 10–20% / 11–21% (abstract, §4.7) | ✓ | ✓ | — | ✓ ("underestimating total cooling") | **OK** |
| "Matched the best cooling that any tested search found" (no "optimum") | ✓ contribution 2, §5.3, conclusion | ✓ | ✓ H4 | ✓ | **OK.** "cooling optimum" occurs 0 times in all files. |
| "We compared the following methods" (5-item list, §3.8) | ✓ | ✓ | — | — | **OK.** "four methods" occurs 0 times. |
| 90–94% of greedy's cooling (§4.5) | ✓ (facts 90–94 at 50–100 trees) | ✓ | — | — | **OK** |
| Sensitivity analysis | ✓ §3.9, §4.6, printed as **Table 5** (file `T7_sensitivity`) | ✓ | — | — | **OK.** Numbers audited by `check_numbers.py` (89 claims). |
| Multi-seed stability | ✓ §4.5 ¶3, §5.5 | ✓ | — | — | **OK** |
| Equity honesty | ✓ abstract ("mainly by cooling the least exposed district less"), §4.5 (19/20), §5.3 | ✓ | ✓ H5 | ✓ | **OK** |
| GNN-for-equity caveat (§5.3) | ✓ "We did not test whether the GNN changes the equity trade-offs…" | ✓ | — | — | **OK** |
| "Can matter as much as" (hedged) | ✓ abstract, conclusion | ✓ | ✓ H1 | ✓ | **OK.** "matters as much" occurs 0 times. |
| Practitioner rule 46–85% / 29–71%, unseeded NSGA-III 60–91% | ✓ (facts match) | ✓ | ✓ H3 | ✓ | **OK** |
| No "v1/v2", "independent verification", "lower-quality input", "decision-grade", "sets a benchmark" | 0 occurrences ("v2" appears only inside the method name GATv2) | 0 | 0 | 0 | **OK** |
| Blind body identical to main | Source diff = exactly the intended removals (comments, PDF author, author block, CRediT, URL) | — | — | — | **OK** |

## 2. Mechanical and submission rules

| Rule | Measured | Status |
|---|---|---|
| Abstract ≤ 250 words (rendered) | **246** (main and blind) | Pass |
| Abstract ≤ 3 acronyms | UTCI, GNN, SOLWEIG-GPU (3) | Pass |
| Keywords ≤ 6 | **6** | Pass |
| Highlights 3–5, ≤ 85 characters | 5 bullets: 64, 79, 78, 83, 69 | Pass. All hedged or numeric; no overclaim. |
| Overfull boxes | 0 in all four documents | Pass |
| Underfull boxes | main.pdf **1** (badness 1728, the GitHub-URL line in Data and code availability); blind, title page and letter 0 | **Micro-fix 2** |
| Broken refs/citations (`??`, `[?]`, `(?)`; undefined or multiply defined labels) | 0 everywhere; BibTeX reports no warnings | Pass |
| Duplicate arXiv IDs | 10 IDs printed, 0 duplicated; `eprint` fields 0 | Pass |
| Double-blind: `manuscript_blind.pdf` | Text, metadata, link targets and 198 inflated streams contain **no** author name, affiliation, city, e-mail, GitHub URL, "corresponding author", CRediT or acknowledgement. Only "grant" appears, in "did not receive any specific grant". PDF author metadata is empty. | **Pass** |
| Title page | Title, 3 authors and affiliation, corresponding author with unhyphenated e-mail, acknowledgements ("None"), CRediT, declarations, data/code with the GitHub URL, AI declaration | Pass |
| Cover letter | Signed by the corresponding author; contact e-mail; article type; findings; limits; originality; competing interests; AI; code link | Pass. "Research Paper" article-type name **unverified**. The letter contains the GitHub URL; it is normally editor-only, but do not upload it as a reviewer-visible file. |
| Reference style | Elsevier Harvard (`elsarticle-harv`); DOI links use legacy `http://dx.doi.org/` | **Unverified** against the guide (APA 7 requested in your checklist). Acceptable if the journal accepts any style at first submission. |

## 3. Remaining micro-issues

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | **Fig. 6 caption says "Orange circles mark the five layouts verified with SOLWEIG-GPU"**, but the figure marks **four** circles (low budget, knee, high budget, max cooling); the fifth verified layout is "no new trees". The text elsewhere says "four optimised layouts". | Fig. 6 caption | "…mark the four optimised layouts verified with SOLWEIG-GPU." |
| 2 | One underfull line (badness 1728) at the long GitHub URL. | main.pdf, Data and code availability | `\usepackage{xurl}` in place of `url`, so the URL can break anywhere. |
| 3 | **SOLWEIG is never spelled out** anywhere in the paper. | §1 ¶3, first mention | "SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry model)". |
| 4 | The abstract says "per-segment correlations of 0.91–0.94"; they are per segment-hour (Table 6 caption, §4.7). | Abstract | "segment-hour correlations" (same word count). The audit anchor in `check_numbers.py` needs the same change. |
| 5 | Results archive (labels, trained models, benchmark outputs) is still "Zenodo on publication; to reviewers via the editor". | Data and code availability | Optional: a private Zenodo reviewer link before upload. |
| 6 | Guide-dependent items: review model (double-blind assumed), APA 7 vs Harvard, graphical abstract, word limit (~9,000 words body), article-type name. | — | You, in the live guide (F12). |

Items 1–4 are editorial slips that a hostile Reviewer #2 would spot, but none changes a result. Items 5–6 are submission logistics.

### Status after the micro-fixes
Items 1–4 have been applied: the Fig. 6 caption now reads "four optimised layouts"; `xurl` is used; SOLWEIG is spelled out at first mention; the abstract says "segment-hour correlations", with the audit anchor updated. After a rebuild, all four documents have 0 overfull and 0 underfull boxes and 0 undefined references. The abstract is still 246 words, and the blind scan is still clean. Items 5–6 remain with the authors. Readiness after the fixes: **94/100**, with the remaining points tied to the unverified guide items and the acknowledged design limits.

## 4. Score rationale (91 before the micro-fixes)

- **+** Every scientific weakness from round 1 is either fixed (R1–R6, R9, DA-1…DA-5) or reported transparently (R7 substance, R8 GNN training seed, DA-6).
- **+** Numbers are machine-audited against the saved results (89 claims). `reproduce.py --tests` passes. The mechanical rules all pass.
- **+** The blind file is verifiably clean at byte level.
- **−3** Four micro-slips (§3, items 1–4).
- **−3** Guide compliance not verifiable (reference style, review model, article type).
- **−3** Residual design limits that reviewers will raise (single SOLWEIG patch, no field data, population proxy, single GNN training run, results archive not yet shared).

**Upload decision:** apply items 1–4, rebuild (`make_blind.py` + the four tectonic builds), confirm F12 in the guide, then upload `manuscript_blind.pdf`, `title_page.pdf`, `highlights.txt`, `cover_letter.pdf` and the figures.
