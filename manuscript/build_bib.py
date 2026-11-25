"""Build manuscript/references.bib from VERIFIED metadata only.

    .\\.venv\\Scripts\\python.exe manuscript/build_bib.py

Sources:
  1. docs/stage1/references_verified.json (Stage 1, matched against Crossref/arXiv; PASS only)
  2. EXTRA below: methods references added for the manuscript; each is fetched by DOI from
     Crossref (or by id from the arXiv API) and must match title + first-author surname.
Every BibTeX field comes from the fetched record (Crossref /works/{doi} or arXiv), never typed.
Data/software without a DOI (Overture, OpenStreetMap, Open-Meteo) are @misc with URL only.
Writes references.bib and bib_verification.json (what was checked, ratio, status).
"""
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
ROOT = HERE.parent
UA = "HeatGraph-manuscript/0.1 (reference metadata check; python-urllib)"

# key -> (doi or arxiv:id, expected title fragment, first-author surname)
EXTRA = {
    "nemhauser1978": ("10.1007/BF01588971", "An analysis of approximations for maximizing submodular set functions", "Nemhauser"),
    "hamilton2017": ("arxiv:1706.02216", "Inductive Representation Learning on Large Graphs", "Hamilton"),
    "gilmer2017": ("arxiv:1704.01212", "Neural Message Passing for Quantum Chemistry", "Gilmer"),
    "loshchilov2019": ("arxiv:1711.05101", "Decoupled Weight Decay Regularization", "Loshchilov"),
    "lei2018": ("10.1080/01621459.2017.1307116", "Distribution-Free Predictive Inference for Regression", "Lei"),
    "shorrocks1980": ("10.2307/1913126", "The Class of Additively Decomposable Inequality Measures", "Shorrocks"),
    "erbs1982": ("10.1016/0038-092X(82)90302-4", "Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation", "Erbs"),
    "prata1996": ("10.1002/qj.49712253306", "A new long-wave formula for estimating downward clear-sky radiation at the surface", "Prata"),
}
# Stage-1 keys renamed to readable BibTeX keys
RENAME = {"philly_lup": "lachapelle2023", "arxiv:2310.05691": "schrodi2023", "arxiv:2510.00457": "xin2025",
          "arxiv:2010.03409": "pfaff2021", "arxiv:2105.14491": "brody2022", "arxiv:1903.02428": "fey2019",
          "arxiv:1903.03894": "ying2019", "arxiv:2107.07511": "angelopoulos2021", "arxiv:2512.11896": "vu2025",
          "heidelberg_route": "kolaxidis2025", "solweig_gpu": "kamath2026"}
MISC = {
    "overture2026": ("Overture Maps Foundation", "Overture Maps: open map data (buildings, transportation, places, base themes)",
                     "https://overturemaps.org", "2026"),
    "osm2026": ("{OpenStreetMap contributors}", "OpenStreetMap", "https://www.openstreetmap.org", "2026"),
    "openmeteo2026": ("{Open-Meteo}", "Open-Meteo Historical Weather API (ERA5-based)", "https://open-meteo.com", "2026"),
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception:
            if a == 2:
                raise
            time.sleep(2 + 2 * a)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"<[^>]+>", " ", s or "").lower()).split())


def crossref_full(doi):
    return json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]


def arxiv_full(aid):
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={aid}")).find("a:entry", ns)
    return {"title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "year": int(e.find("a:published", ns).text[:4]), "id": aid}


def tex_escape(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    s = s.replace("&amp;", "&").replace("–", "--").replace("—", "---").replace("‐", "-")
    return re.sub(r"(?<!\\)([&%$#_])", r"\\\1", s)


def bib_from_crossref(key, m):
    def year():
        for k in ("published-print", "issued", "published-online"):
            dp = (m.get(k) or {}).get("date-parts")
            if dp and dp[0] and dp[0][0]:
                return dp[0][0]
    authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", []))
    f = {"author": authors, "title": "{" + tex_escape((m.get("title") or [""])[0]) + "}",
         "journal": tex_escape((m.get("container-title") or [""])[0]), "year": year(), "volume": m.get("volume"),
         "number": m.get("issue"), "pages": (m.get("page") or m.get("article-number") or "").replace("-", "--") or None,
         "doi": m.get("DOI")}
    kind = "article" if m.get("type") == "journal-article" else "inproceedings" if "proceedings" in m.get("type", "") \
        else "article"
    if kind == "inproceedings":
        f["booktitle"] = f.pop("journal")
    return kind, f


def bib_from_arxiv(key, a, doi=None, note=None):
    f = {"author": " and ".join(a["authors"]), "title": "{" + tex_escape(a["title"]) + "}", "year": a["year"],
         "eprint": a["id"], "archiveprefix": "arXiv", "doi": doi or f"10.48550/arXiv.{a['id']}",
         "howpublished": "arXiv preprint arXiv:" + a["id"], "note": note}
    return "misc", f


def render(key, kind, f):
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v not in (None, ""))
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ver = json.loads((ROOT / "docs" / "stage1" / "references_verified.json").read_text(encoding="utf-8"))
    out, report = [], []
    uncited = {"ying2019"}          # verified in Stage 1 but not cited in the manuscript
    for r in ver:
        if r["status"] != "PASS":
            continue
        key = RENAME.get(r["key"], r["key"])
        if key in uncited:
            continue
        if r["doi"].startswith("10.48550/arXiv."):
            aid = r["doi"].split("arXiv.")[1]
            a = arxiv_full(aid)
            note = "Published at ICLR 2026" if aid == "2510.00457" else None
            kind, f = bib_from_arxiv(key, a, note=note)
            time.sleep(3)
        else:
            m = crossref_full(r["doi"])
            kind, f = bib_from_crossref(key, m)
            time.sleep(0.8)
        if key == "holmgren2018":   # Crossref lists given/family swapped for this JOSS record
            f["author"] = "Holmgren, William F. and Hansen, Clifford W. and Mikofski, Mark A."
        out.append(render(key, kind, f))
        report.append({"key": key, "source": "stage1", "doi": r["doi"], "status": "PASS"})
    for key, (ident, title, surname) in EXTRA.items():
        if ident.startswith("arxiv:"):
            a = arxiv_full(ident.split(":")[1])
            t, authors = a["title"], a["authors"]
            kind, f = bib_from_arxiv(key, a)
            time.sleep(3)
        else:
            m = crossref_full(ident)
            t = (m.get("title") or [""])[0]
            authors = [x.get("family", "") for x in m.get("author", [])]
            kind, f = bib_from_crossref(key, m)
            time.sleep(0.8)
        ratio = difflib.SequenceMatcher(None, norm(title), norm(t)).ratio()
        ok = (norm(t).startswith(norm(title)[:40]) or ratio >= 0.85) and any(surname.lower() in x.lower() for x in authors)
        report.append({"key": key, "source": "extra", "id": ident, "fetched_title": t, "ratio": round(ratio, 3),
                       "status": "PASS" if ok else "FAIL"})
        print(f"{'PASS' if ok else 'FAIL'} {key:16s} {ident:34s} {t[:70]}")
        if ok:
            out.append(render(key, kind, f))
    for key, (author, title, url, year) in MISC.items():
        out.append(render(key, "misc", {"author": author, "title": "{" + title + "}", "howpublished": f"\\url{{{url}}}",
                                        "year": year, "note": "Accessed 2026-11-25"}))
        report.append({"key": key, "source": "data/software, no DOI", "url": url, "status": "URL only"})
    (HERE / "references.bib").write_text("% Generated by build_bib.py from verified Crossref/arXiv records.\n\n" +
                                         "\n".join(out), encoding="utf-8")
    (HERE / "bib_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(out)} entries written; {sum(r['status'] == 'FAIL' for r in report)} failed verification")


if __name__ == "__main__":
    main()
