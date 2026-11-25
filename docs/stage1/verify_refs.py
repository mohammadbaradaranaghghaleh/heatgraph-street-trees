"""Verify candidate references against Crossref (journal/proceedings) and arXiv (preprints).

Stdlib only, small JSON/XML requests (no file downloads). Every candidate is matched by
title + first-author surname; a reference PASSES only when the returned title is a close
match (difflib ratio >= 0.85) AND the first-author surname appears in the returned authors.
Anything else is FAIL and must not be cited ("gray zone = FAIL").

    py -3 docs/stage1/verify_refs.py      ->  references_verified.json + references_verified.md
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "HeatGraph-lit-review/0.1 (academic reference verification; python-urllib)"

# (key, theme, first-author surname, approx year, title)  -- or ("arxiv:<id>", ...)
CANDIDATES = [
    # T1 pedestrian thermal comfort: Tmrt / UTCI / radiation modelling
    ("lindberg2008", "T1", "Lindberg", 2008, "SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings"),
    ("lindberg2011", "T1", "Lindberg", 2011, "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation"),
    ("lindberg2018", "T1", "Lindberg", 2018, "Urban Multi-scale Environmental Predictor (UMEP): An integrated tool for city-based climate services"),
    ("thorsson2007", "T1", "Thorsson", 2007, "Different methods for estimating the mean radiant temperature in an outdoor urban setting"),
    ("brode2012", "T1", "Bröde", 2012, "Deriving the operational procedure for the Universal Thermal Climate Index (UTCI)"),
    ("tartarini2020", "T1", "Tartarini", 2020, "pythermalcomfort: A Python package for thermal comfort research"),
    ("ratti2004", "T1", "Ratti", 2004, "Raster analysis of urban form"),
    ("middel2021", "T1", "Middel", 2021, "50 Grades of Shade"),
    ("middel2016", "T1", "Middel", 2016, "Impact of shade on outdoor thermal comfort—a seasonal field study in Tempe, Arizona"),
    ("konarska2014", "T1", "Konarska", 2014, "Transmissivity of solar radiation through crowns of single urban trees—application for outdoor thermal comfort modelling"),
    ("holmgren2018", "T1", "Holmgren", 2018, "pvlib python: a python package for modeling solar energy systems"),
    ("reda2004", "T1", "Reda", 2004, "Solar position algorithm for solar radiation applications"),
    # T2 trees, cooling and placement optimisation
    ("bowler2010", "T2", "Bowler", 2010, "Urban greening to cool towns and cities: A systematic review of the empirical evidence"),
    ("rahman2020", "T2", "Rahman", 2020, "Traits of trees for cooling urban heat islands: A meta-analysis"),
    ("ziter2019", "T2", "Ziter", 2019, "Scale-dependent interactions between tree canopy cover and impervious surfaces reduce daytime urban heat during summer"),
    ("schwaab2021", "T2", "Schwaab", 2021, "The role of urban trees in reducing land surface temperatures in European cities"),
    ("lee2016", "T2", "Lee", 2016, "Contribution of trees and grasslands to the mitigation of human heat stress in a residential district of Freiburg, Southwest Germany"),
    ("zhao2018", "T2", "Zhao", 2018, "Impact of tree locations and arrangements on outdoor microclimates and human thermal comfort in an urban residential environment"),
    ("philly_lup", "T2", "", 2023, "Maximizing pedestrian radiative cooling benefit per street tree"),
    ("arxiv:2310.05691", "T2", "Schrodi", 2023, "Climate-sensitive Urban Planning through Optimization of Tree Placements"),
    ("kim2025", "T2", "Kim", 2025, "Deep learning-based spatial optimization of green and cool roof implementation for urban heat mitigation"),
    # T3 neural / graph surrogates
    ("briegel2023", "T3", "Briegel", 2023, "Modelling long-term thermal comfort conditions in urban environments using a deep convolutional encoder-decoder as a computationally efficient alternative to physics-based simulation"),
    ("briegel2024", "T3", "Briegel", 2024, "High-resolution multi-scaling of outdoor human thermal comfort and its intra-urban variability based on machine learning"),
    ("yu2024", "T3", "Yu", 2024, "Street-level temperature estimation using graph neural networks: Performance, feature embedding and interpretability"),
    ("arxiv:2510.00457", "T3", "Xin", 2025, "UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction"),
    ("arxiv:2010.03409", "T3", "Pfaff", 2020, "Learning Mesh-Based Simulation with Graph Networks"),
    ("arxiv:2105.14491", "T3", "Brody", 2021, "How Attentive are Graph Attention Networks?"),
    ("arxiv:1903.02428", "T3", "Fey", 2019, "Fast Graph Representation Learning with PyTorch Geometric"),
    ("arxiv:1903.03894", "T3", "Ying", 2019, "GNNExplainer: Generating Explanations for Graph Neural Networks"),
    ("arxiv:2107.07511", "T3", "Angelopoulos", 2021, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"),
    ("deb2014", "T3", "Deb", 2014, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints"),
    ("blank2020", "T3", "Blank", 2020, "pymoo: Multi-Objective Optimization in Python"),
    # T4 pedestrian exposure, networks, routing
    ("sevtsuk2021", "T4", "Sevtsuk", 2021, "Estimating Pedestrian Flows on Street Networks"),
    ("boeing2017", "T4", "Boeing", 2017, "OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks"),
    ("kuras2017", "T4", "Kuras", 2017, "Opportunities and Challenges for Personal Heat Exposure Research"),
    ("heidelberg_route", "T4", "", 2024, "Mitigating heat stress by reducing solar exposure in pedestrian routing"),
    ("arxiv:2512.11896", "T4", "", 2025, "Hot Hẻm"),
    # T5 heat and canopy equity
    ("hsu2021", "T5", "Hsu", 2021, "Disproportionate exposure to urban heat island intensity across major US cities"),
    ("mcdonald2021", "T5", "McDonald", 2021, "The tree cover and temperature disparity in US urbanized areas: Quantifying the association with income across 5,723 communities"),
    ("locke2021", "T5", "Locke", 2021, "Residential housing segregation and urban tree canopy in 37 US Cities"),
    ("schwarz2015", "T5", "Schwarz", 2015, "Trees Grow on Money: Urban Tree Canopy Cover and Environmental Justice"),
    ("chakraborty2019", "T5", "Chakraborty", 2019, "Disproportionately higher exposure to urban heat in lower-income neighborhoods: a multi-city perspective"),
    ("benz2021", "T5", "Benz", 2021, "Widespread Race and Class Disparities in Surface Urban Heat Extremes Across the United States"),
    # T6 open data
    ("tolan2024", "T6", "Tolan", 2024, "Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar"),
    ("peters2022", "T6", "Peters", 2022, "Automated 3D reconstruction of LoD2 and LoD1 models for all 10 million buildings of the Netherlands"),
    ("demuzere2022", "T6", "Demuzere", 2022, "A global map of local climate zones to support earth system modelling and urban-scale environmental science"),
    ("stewart2012", "T6", "Stewart", 2012, "Local Climate Zones for Urban Temperature Studies"),
    ("hersbach2020", "T6", "Hersbach", 2020, "The ERA5 global reanalysis"),
    # added in pass 2
    ("sanchez2019", "T5", "Sanchez", 2019, "Cooling Detroit: A socio-spatial analysis of equity in green roofs as an urban heat island mitigation strategy"),
    ("solweig_gpu", "T1", "Kamath", 2026, "SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins"),
]

# Pass 2: preprint DOIs returned by the free-text search are replaced by the journal
# version of record; each pin is still fetched and checked (title + first author).
DOI_PINS = {
    "briegel2023": "10.1016/j.uclim.2022.101359",
    "briegel2024": "10.5194/gmd-17-1667-2024",
    "boeing2017": "10.1016/j.compenvurbsys.2017.05.004",
    "locke2021": "10.1038/s42949-021-00022-0",
    "benz2021": "10.1029/2021EF002016",
    "demuzere2022": "10.5194/essd-14-3835-2022",
    "hersbach2020": "10.1002/qj.3803",
    "kim2025": "10.1016/j.jenvman.2025.125398",
}
# Free-text search restricted to journal articles (avoid preprint servers).
JOURNAL_ONLY = {"yu2024", "heidelberg_route"}
# Queries that describe a paper instead of quoting its title: never auto-PASS,
# the returned record is printed for manual confirmation.
DESCRIPTIVE: set[str] = set()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + 2 * attempt)
    return b""


def _norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _record(key, theme, author, title, it, ratio, note=""):
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in it.get("author", [])]
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    surname_ok = (not author) or any(author.lower() in a.lower() for a in authors)
    ok = ratio >= 0.85 and surname_ok and key not in DESCRIPTIVE
    return {"key": key, "theme": theme, "status": "PASS" if ok else ("CHECK" if key in DESCRIPTIVE else "FAIL"),
            "match_ratio": round(ratio, 3), "doi": it.get("DOI"), "title": (it.get("title") or [""])[0],
            "authors": authors, "year": yr, "venue": (it.get("container-title") or [""])[0],
            "type": it.get("type"), "note": note}


def crossref_doi(key, theme, author, year, title):
    it = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(DOI_PINS[key])))["message"]
    t = (it.get("title") or [""])[0]
    # pinned DOI: accept a looser title ratio (subtitle/punctuation variants) when the surname matches
    r = max(_ratio(title, t), 1.0 if _norm(t).startswith(_norm(title)[:60]) else 0.0)
    return _record(key, theme, author, title, it, r, note="pinned version-of-record DOI")


def crossref(key, theme, author, year, title):
    if key in DOI_PINS:
        return crossref_doi(key, theme, author, year, title)
    q = {"query.bibliographic": f"{title} {author}".strip(), "rows": 5,
         "select": "DOI,title,author,issued,container-title,type"}
    if key in JOURNAL_ONLY:
        q["filter"] = "type:journal-article"
    data = json.loads(_get("https://api.crossref.org/works?" + urllib.parse.urlencode(q)))
    best, best_r = None, 0.0
    for it in data["message"]["items"]:
        t = (it.get("title") or [""])[0]
        r = _ratio(title, t)
        if r > best_r:
            best, best_r = it, r
    if not best:
        return {"key": key, "status": "FAIL", "reason": "no Crossref hit"}
    rec = _record(key, theme, author, title, best, best_r)
    if rec["status"] == "PASS" and rec["type"] == "posted-content":
        rec["status"], rec["note"] = "FAIL", "preprint DOI returned; pin the version of record"
    return rec


def arxiv(key, theme, author, year, title):
    aid = key.split(":", 1)[1]
    xml = _get(f"https://export.arxiv.org/api/query?id_list={aid}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(xml).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return {"key": key, "status": "FAIL", "reason": "arXiv id not found"}
    t = " ".join(e.find("a:title", ns).text.split())
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    pub = e.find("a:published", ns).text[:4]
    if len(title.split()) > 3:
        r = _ratio(title, t)
    else:  # short title stub: compare accent-stripped prefixes
        import unicodedata
        strip = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        r = 1.0 if strip(t).startswith(strip(title)[:5]) else 0.0
    surname_ok = (not author) or any(author.lower() in a.lower() for a in authors)
    return {"key": key, "theme": theme, "status": "PASS" if (r >= 0.85 and surname_ok) else "FAIL",
            "match_ratio": round(r, 3), "doi": f"10.48550/arXiv.{aid}", "title": t, "authors": authors,
            "year": int(pub), "venue": "arXiv", "type": "preprint"}


def main() -> int:
    for s in (sys.stdout,):
        s.reconfigure(encoding="utf-8", errors="replace")
    out = []
    for c in CANDIDATES:
        fn = arxiv if c[0].startswith("arxiv:") else crossref
        try:
            res = fn(*c)
        except Exception as ex:
            res = {"key": c[0], "status": "FAIL", "reason": f"{type(ex).__name__}: {ex}"}
        res["candidate_title"] = c[4]
        out.append(res)
        print(f"{res['status']:4} {res.get('match_ratio', '-')!s:>5}  {c[0]:<18} {res.get('doi', '')}  | {res.get('title', res.get('reason', ''))[:90]}")
        time.sleep(1.0 if fn is crossref else 3.0)  # Crossref/arXiv etiquette
    (HERE / "references_verified.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n = sum(r["status"] == "PASS" for r in out)
    print(f"\nPASS {n}/{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
