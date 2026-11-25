"""Render references_verified.json -> references_verified.md (APA-like, grouped by theme).

Only PASS records are rendered; metadata comes from Crossref/arXiv, never typed by hand.
"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THEMES = {
    "T1": "T1 — Pedestrian thermal comfort: Tmrt, UTCI and radiation modelling",
    "T2": "T2 — Trees, cooling and placement optimisation",
    "T3": "T3 — Neural and graph surrogates, explanation, uncertainty, optimisation",
    "T4": "T4 — Pedestrian networks, flows, exposure and routing",
    "T5": "T5 — Heat and canopy equity",
    "T6": "T6 — Open data foundations",
}


def apa_authors(authors):
    names = []
    for a in authors:
        if "," in a:
            fam, giv = [s.strip() for s in a.split(",", 1)]
            initials = " ".join(f"{p[0]}." for p in giv.replace("-", " ").split() if p)
            names.append(f"{fam}, {initials}".strip(", "))
        else:  # arXiv "Given Family"
            parts = a.split()
            names.append(f"{parts[-1]}, {' '.join(p[0] + '.' for p in parts[:-1])}".strip(", "))
    if len(names) > 20:
        names = names[:19] + ["...", names[-1]]
    return ", ".join(names[:-1]) + (", & " if len(names) > 1 else "") + names[-1] if names else "[no authors]"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    refs = [r for r in json.loads((HERE / "references_verified.json").read_text(encoding="utf-8"))
            if r["status"] == "PASS"]
    lines = ["# Verified reference corpus (HeatGraph, Stage 1)", "",
             f"{len(refs)} references. Each was matched by title and first-author surname against "
             "Crossref or the arXiv API (`verify_refs.py`); DOIs are versions of record unless the "
             "work exists only as a preprint. Regenerate with `py -3 docs/stage1/build_bib.py`.", ""]
    for code, label in THEMES.items():
        group = sorted((r for r in refs if r.get("theme") == code), key=lambda r: (r["authors"] or [""])[0])
        lines += [f"## {label} ({len(group)})", ""]
        for r in group:
            venue = f" *{html.unescape(r['venue'])}*." if r.get("venue") else ""
            kind = " [preprint]" if r.get("type") in ("preprint", "posted-content") else ""
            lines.append(f"- **[{r['key']}]** {apa_authors(r['authors'])} ({r['year']}). {r['title']}.{venue}"
                         f" https://doi.org/{r['doi']}{kind}")
        lines.append("")
    (HERE / "references_verified.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote references_verified.md ({len(refs)} refs)")


if __name__ == "__main__":
    main()
